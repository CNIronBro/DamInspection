"""MySQL 数据库连接池 + CRUD 操作"""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import mysql.connector
from mysql.connector import pooling

# 加载 .env 文件（从项目根目录向上查找）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# 连接池配置
_pool: Optional[pooling.MySQLConnectionPool] = None

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "127.0.0.1"),
    "port": int(os.environ.get("DB_PORT", "3306")),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "dam_inspection"),
    "charset": "utf8mb4",
    "autocommit": True,
}

IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", "/root/work/pic"))

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS detection_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_name VARCHAR(255) NOT NULL,
    image_width INT,
    image_height INT,
    image_path VARCHAR(512),
    model_id VARCHAR(64) DEFAULT 'segformer-b2',
    mm_per_px FLOAT DEFAULT 1.0,
    total_instances INT DEFAULT 0,
    by_class_json JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS detection_instances (
    id INT AUTO_INCREMENT PRIMARY KEY,
    record_id INT NOT NULL,
    class_name VARCHAR(64),
    class_name_cn VARCHAR(64),
    instance_id INT,
    source ENUM('auto','manual') DEFAULT 'auto',
    area_mm2 DOUBLE,
    length_mm DOUBLE,
    width_mean_mm DOUBLE,
    width_max_mm DOUBLE,
    width_p95_mm DOUBLE,
    eq_diameter_mm DOUBLE,
    major_axis_mm DOUBLE,
    minor_axis_mm DOUBLE,
    bbox_json JSON,
    contour_json JSON,
    centroid_json JSON,
    FOREIGN KEY (record_id) REFERENCES detection_records(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def _get_pool() -> pooling.MySQLConnectionPool:
    global _pool
    if _pool is None:
        _pool = pooling.MySQLConnectionPool(
            pool_name="dam_pool",
            pool_size=5,
            **DB_CONFIG,
        )
    return _pool


@contextmanager
def get_conn():
    """获取数据库连接（上下文管理器）"""
    conn = _get_pool().get_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """初始化数据库表结构 + 图片目录"""
    # 创建图片存储目录
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # 先确保数据库存在
    cfg = {k: v for k, v in DB_CONFIG.items() if k != "database"}
    conn = mysql.connector.connect(**cfg)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}` CHARACTER SET utf8mb4")
    cursor.close()
    conn.close()

    # 创建表（先检查旧表结构，如有 image_base64 字段则重建）
    with get_conn() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SHOW COLUMNS FROM detection_records LIKE 'image_base64'")
            if cursor.fetchone():
                print("[INFO] Migrating: dropping old detection_records table (had image_base64)")
                cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
                cursor.execute("DROP TABLE IF EXISTS detection_instances")
                cursor.execute("DROP TABLE IF EXISTS detection_records")
                cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        except Exception:
            pass  # 表不存在，首次运行

        for stmt in SCHEMA_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                cursor.execute(stmt)
        cursor.close()


# --------------- Records CRUD ---------------

def save_record(record: Dict[str, Any], instances: List[Dict[str, Any]]) -> int:
    """保存一条检测记录及其所有实例，返回 record_id"""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO detection_records
               (image_name, image_width, image_height, image_path,
                model_id, mm_per_px, total_instances, by_class_json)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                record["image_name"],
                record.get("image_width"),
                record.get("image_height"),
                record.get("image_path"),
                record.get("model_id", "segformer-b2"),
                record.get("mm_per_px", 1.0),
                record.get("total_instances", 0),
                json.dumps(record.get("by_class", {}), ensure_ascii=False),
            ),
        )
        record_id = cursor.lastrowid

        for inst in instances:
            cursor.execute(
                """INSERT INTO detection_instances
                   (record_id, class_name, class_name_cn, instance_id, source,
                    area_mm2, length_mm, width_mean_mm, width_max_mm, width_p95_mm,
                    eq_diameter_mm, major_axis_mm, minor_axis_mm,
                    bbox_json, contour_json, centroid_json)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    record_id,
                    inst.get("class_name"),
                    inst.get("class_name_cn"),
                    inst.get("instance_id"),
                    inst.get("source", "auto"),
                    inst.get("area_mm2"),
                    inst.get("length_mm"),
                    inst.get("width_mean_mm"),
                    inst.get("width_max_mm"),
                    inst.get("width_p95_mm"),
                    inst.get("eq_diameter_mm"),
                    inst.get("major_axis_mm"),
                    inst.get("minor_axis_mm"),
                    json.dumps(inst.get("bbox"), ensure_ascii=False),
                    json.dumps(inst.get("contour"), ensure_ascii=False),
                    json.dumps(inst.get("centroid"), ensure_ascii=False),
                ),
            )
        cursor.close()
        conn.commit()
        return record_id


def update_record(record_id: int, record: Dict[str, Any], instances: List[Dict[str, Any]]) -> int:
    """更新已有记录：替换元数据 + 删除旧实例 + 插入新实例"""
    with get_conn() as conn:
        cursor = conn.cursor()
        # 更新主记录
        cursor.execute(
            """UPDATE detection_records SET
               image_name=%s, image_width=%s, image_height=%s, image_path=%s,
               model_id=%s, mm_per_px=%s, total_instances=%s, by_class_json=%s
               WHERE id=%s""",
            (
                record["image_name"],
                record.get("image_width"),
                record.get("image_height"),
                record.get("image_path"),
                record.get("model_id", "segformer-b2"),
                record.get("mm_per_px", 1.0),
                record.get("total_instances", 0),
                json.dumps(record.get("by_class", {}), ensure_ascii=False),
                record_id,
            ),
        )
        # 删除旧实例
        cursor.execute("DELETE FROM detection_instances WHERE record_id = %s", (record_id,))

        # 插入新实例
        for inst in instances:
            cursor.execute(
                """INSERT INTO detection_instances
                   (record_id, class_name, class_name_cn, instance_id, source,
                    area_mm2, length_mm, width_mean_mm, width_max_mm, width_p95_mm,
                    eq_diameter_mm, major_axis_mm, minor_axis_mm,
                    bbox_json, contour_json, centroid_json)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    record_id,
                    inst.get("class_name"),
                    inst.get("class_name_cn"),
                    inst.get("instance_id"),
                    inst.get("source", "auto"),
                    inst.get("area_mm2"),
                    inst.get("length_mm"),
                    inst.get("width_mean_mm"),
                    inst.get("width_max_mm"),
                    inst.get("width_p95_mm"),
                    inst.get("eq_diameter_mm"),
                    inst.get("major_axis_mm"),
                    inst.get("minor_axis_mm"),
                    json.dumps(inst.get("bbox"), ensure_ascii=False),
                    json.dumps(inst.get("contour"), ensure_ascii=False),
                    json.dumps(inst.get("centroid"), ensure_ascii=False),
                ),
            )
        cursor.close()
        conn.commit()
        return record_id


def list_records(page: int = 1, size: int = 20) -> Dict[str, Any]:
    """分页查询记录列表（不含实例详情）"""
    offset = (page - 1) * size
    with get_conn() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT id, image_name, image_width, image_height,
                      model_id, mm_per_px, total_instances, by_class_json, created_at
               FROM detection_records
               ORDER BY created_at DESC
               LIMIT %s OFFSET %s""",
            (size, offset),
        )
        rows = cursor.fetchall()

        cursor.execute("SELECT COUNT(*) AS cnt FROM detection_records")
        total = cursor.fetchone()["cnt"]
        cursor.close()

        for row in rows:
            if row.get("by_class_json"):
                row["by_class"] = json.loads(row.pop("by_class_json"))
            if row.get("created_at"):
                row["created_at"] = row["created_at"].isoformat()

        return {"records": rows, "total": total, "page": page, "size": size}


def get_record(record_id: int) -> Optional[Dict[str, Any]]:
    """获取单条记录详情（含实例列表）"""
    with get_conn() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM detection_records WHERE id = %s", (record_id,))
        record = cursor.fetchone()
        if not record:
            cursor.close()
            return None

        if record.get("by_class_json"):
            record["by_class"] = json.loads(record.pop("by_class_json"))
        if record.get("created_at"):
            record["created_at"] = record["created_at"].isoformat()

        cursor.execute(
            "SELECT * FROM detection_instances WHERE record_id = %s ORDER BY class_name, instance_id",
            (record_id,),
        )
        instances = cursor.fetchall()
        cursor.close()

        for inst in instances:
            for field in ("bbox_json", "contour_json", "centroid_json"):
                if inst.get(field):
                    key = field.replace("_json", "")
                    inst[key] = json.loads(inst.pop(field))

        record["instances"] = instances
        return record


def delete_record(record_id: int) -> bool:
    """删除一条记录（含图片文件）"""
    # 先获取图片路径
    image_path = None
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM detection_records WHERE id = %s", (record_id,))
        row = cursor.fetchone()
        if row:
            image_path = row[0]
        cursor.close()

    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM detection_records WHERE id = %s", (record_id,))
        deleted = cursor.rowcount > 0
        cursor.close()
        conn.commit()

    # 删除图片文件
    if deleted and image_path:
        try:
            p = Path(image_path)
            if p.exists():
                p.unlink()
        except OSError:
            pass

    return deleted


# --------------- 统计查询 ---------------

def get_stats_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    class_name: Optional[str] = None,
) -> Dict[str, Any]:
    """聚合统计：按类型、按时间"""
    with get_conn() as conn:
        cursor = conn.cursor(dictionary=True)

        # 日期筛选条件
        where_parts = []
        params = []
        if start_date:
            where_parts.append("r.created_at >= %s")
            params.append(start_date)
        if end_date:
            where_parts.append("r.created_at <= %s")
            params.append(end_date + " 23:59:59")
        if class_name:
            where_parts.append("i.class_name = %s")
            params.append(class_name)

        where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        # 总记录数
        cursor.execute("SELECT COUNT(*) AS cnt FROM detection_records")
        total_records = cursor.fetchone()["cnt"]

        # 按类型统计实例数
        cursor.execute(
            f"""SELECT i.class_name, i.class_name_cn, COUNT(*) AS count
                FROM detection_instances i
                JOIN detection_records r ON i.record_id = r.id
                {where_sql}
                GROUP BY i.class_name, i.class_name_cn""",
            params,
        )
        by_class = cursor.fetchall()

        # 按日期统计（每天的实例数）
        cursor.execute(
            f"""SELECT DATE(r.created_at) AS date, COUNT(*) AS count
                FROM detection_instances i
                JOIN detection_records r ON i.record_id = r.id
                {where_sql}
                GROUP BY DATE(r.created_at)
                ORDER BY date""",
            params,
        )
        by_date = cursor.fetchall()
        for row in by_date:
            if row.get("date"):
                row["date"] = str(row["date"])

        # 按日期+类型统计
        cursor.execute(
            f"""SELECT DATE(r.created_at) AS date, i.class_name, COUNT(*) AS count
                FROM detection_instances i
                JOIN detection_records r ON i.record_id = r.id
                {where_sql}
                GROUP BY DATE(r.created_at), i.class_name
                ORDER BY date""",
            params,
        )
        by_date_class = cursor.fetchall()
        for row in by_date_class:
            if row.get("date"):
                row["date"] = str(row["date"])

        cursor.close()

        return {
            "total_records": total_records,
            "by_class": by_class,
            "by_date": by_date,
            "by_date_class": by_date_class,
        }
