# MySQL 安装与配置指南（Ubuntu 服务器）

## 1. 安装 MySQL

```bash
apt update && apt install -y mysql-server
```

## 2. 启动 MySQL

### 如果系统有 systemd（非容器环境）

```bash
systemctl start mysql
systemctl enable mysql  # 开机自启
```

### 如果没有 systemd（容器环境）

```bash
service mysql start
```

如果报 `su: warning: cannot change directory to /nonexistent`，执行：

```bash
usermod -d /var/lib/mysql mysql
service mysql start
```

验证是否运行：

```bash
mysqladmin ping
# 应输出: mysqld is alive
```

## 3. 设置 root 密码

```bash
mysql -u root
```

进入 MySQL 命令行后执行（将 `<YOUR_PASSWORD>` 替换为你自己的强密码）：

```sql
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '<YOUR_PASSWORD>';
FLUSH PRIVILEGES;
EXIT;
```

> 密码必须满足强度要求（至少 8 位，含大小写、数字、特殊字符）。

## 4. 创建数据库

```bash
mysql -u root -p
```

输入密码后执行：

```sql
CREATE DATABASE IF NOT EXISTS dam_inspection CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
EXIT;
```

## 5. 验证连接

```bash
mysql -u root -p -e "USE dam_inspection; SELECT 1;"
```

输出 `1` 即表示成功。

## 6. 项目配置

项目通过 `.env` 文件读取数据库配置。复制模板并填入你的密码：

```bash
cp .env.example .env
# 编辑 .env，填入 DB_PASSWORD 等实际值
```

`.env` 文件字段：

| 字段 | 说明 | 默认值 |
|------|------|--------|
| DB_HOST | MySQL 地址 | 127.0.0.1 |
| DB_PORT | MySQL 端口 | 3306 |
| DB_USER | MySQL 用户 | root |
| DB_PASSWORD | MySQL 密码 | （必填） |
| DB_NAME | 数据库名 | dam_inspection |
| IMAGE_DIR | 图片存储目录 | /root/work/pic |

表会在后端首次启动时自动创建（`init_db()`），无需手动建表。

## 7. 安装 Python 依赖

```bash
pip install -r requirements-web.txt
```

## 8. 防火墙（如需远程访问）

仅本机访问可跳过。如需远程连接：

```bash
# 编辑 MySQL 配置
sed -i 's/bind-address.*/bind-address = 0.0.0.0/' /etc/mysql/mysql.conf.d/mysqld.cnf
service mysql restart

# 授权远程用户（将 <YOUR_PASSWORD> 替换为实际密码）
mysql -u root -p -e "CREATE USER 'dam'@'%' IDENTIFIED WITH mysql_native_password BY '<YOUR_PASSWORD>'; GRANT ALL ON dam_inspection.* TO 'dam'@'%'; FLUSH PRIVILEGES;"
```

## 常见问题

**Q: `Can't connect to local MySQL server through socket`**
A: MySQL 没有启动，执行 `service mysql start`

**Q: `Access denied for user 'root'@'localhost'`**
A: 密码错误或未设置，跳过权限启动重置：
```bash
service mysql stop
mysqld_safe --skip-grant-tables &
mysql -u root -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '<YOUR_PASSWORD>'; FLUSH PRIVILEGES;"
killall mysqld
service mysql start
```
