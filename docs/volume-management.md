# Volume Management Guide

## Overview

Cosmos Coherence uses Docker volumes for persistent data storage, ensuring data survives container restarts and updates. This guide covers volume configuration, backup, and restoration procedures.

## Volume Architecture

### Named Volumes

The application uses named Docker volumes for critical data:

- **postgres-data**: PostgreSQL database files
- **redis-data**: Redis cache persistence
- **api-cache**: API service cache data
- **dashboard-cache**: Dashboard service cache data
- **nginx-cache**: Nginx proxy cache (production)
- **certbot-www**: Let's Encrypt webroot (production)
- **certbot-conf**: SSL certificates (production)

### Bind Mounts

Development environment uses bind mounts for:

- **./src:/app/src**: Source code (hot-reload enabled)
- **./configs:/app/configs**: Configuration files
- **./tests:/app/tests**: Test files
- **./data:/app/data**: Local data directory

## Data Persistence Strategy

### Database (PostgreSQL)

- **Volume**: `postgres-data`
- **Mount Point**: `/var/lib/postgresql/data`
- **Backup**: Regular SQL dumps via `pg_dump`
- **Restore**: Via `psql` restore command

### Cache (Redis)

- **Volume**: `redis-data`
- **Mount Point**: `/data`
- **Persistence**: AOF (Append Only File) enabled
- **Backup**: RDB snapshots + AOF files

### Application Data

- **Local Directory**: `./data/`
- **Structure**:
  ```
  data/
  ├── outputs/     # Analysis results
  ├── cache/       # Application cache
  └── results/     # Experiment results
  ```

## Backup Procedures

### Automated Backup

Run the backup script to create a complete system backup:

```bash
./scripts/backup.sh
```

This creates a timestamped backup containing:
- Database dump (compressed)
- Redis dump
- All Docker volumes
- Configuration files
- Backup metadata

### Manual Backup

#### Database Backup

```bash
# Backup PostgreSQL
docker exec cosmos-db pg_dump -U cosmos cosmos_coherence | gzip > backup.sql.gz

# Backup with timestamp
docker exec cosmos-db pg_dump -U cosmos cosmos_coherence | gzip > "backup_$(date +%Y%m%d_%H%M%S).sql.gz"
```

#### Redis Backup

```bash
# Trigger background save
docker exec cosmos-redis redis-cli BGSAVE

# Copy dump file
docker cp cosmos-redis:/data/dump.rdb redis_backup.rdb
```

#### Volume Backup

```bash
# Backup a specific volume
docker run --rm \
  -v cosmos-coherence_postgres-data:/source:ro \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/postgres-data.tar.gz -C /source .
```

## Restore Procedures

### Automated Restore

Use the restore script to restore from a backup:

```bash
# List available backups
./scripts/restore.sh

# Restore specific backup
./scripts/restore.sh 20240101_120000.tar.gz
```

### Manual Restore

#### Database Restore

```bash
# Stop services
docker compose down

# Start only database
docker compose up -d db

# Restore from backup
gunzip < backup.sql.gz | docker exec -i cosmos-db psql -U cosmos cosmos_coherence

# Restart all services
docker compose up -d
```

#### Redis Restore

```bash
# Stop Redis
docker compose stop redis

# Copy dump file
docker cp redis_backup.rdb cosmos-redis:/data/dump.rdb

# Start Redis
docker compose start redis
```

#### Volume Restore

```bash
# Restore a specific volume
docker run --rm \
  -v cosmos-coherence_postgres-data:/target \
  -v $(pwd)/backups:/backup:ro \
  alpine tar xzf /backup/postgres-data.tar.gz -C /target
```

## Volume Management Commands

### List Volumes

```bash
# List all project volumes
docker volume ls | grep cosmos-coherence

# Inspect volume details
docker volume inspect cosmos-coherence_postgres-data
```

### Clean Volumes

```bash
# Remove all project volumes (WARNING: Data loss!)
docker compose down -v

# Remove specific volume
docker volume rm cosmos-coherence_api-cache

# Prune unused volumes
docker volume prune
```

### Volume Size Management

```bash
# Check volume sizes
docker system df -v | grep cosmos-coherence

# Check specific volume size
docker run --rm -v cosmos-coherence_postgres-data:/data alpine du -sh /data
```

## Development vs Production

### Development

- Uses bind mounts for source code (hot-reload)
- Relaxed permissions (read-write mounts)
- Local file access for debugging
- Cache volumes can be safely cleared

### Production

- Read-only mounts for configuration
- Strict permissions (non-root user)
- Named volumes only (no bind mounts)
- Backup volumes for data protection
- SSL certificate volumes

## Best Practices

### Regular Backups

1. **Schedule automated backups**:
   ```bash
   # Add to crontab
   0 2 * * * /path/to/cosmos-coherence/scripts/backup.sh
   ```

2. **Retain backups for 7 days** (configurable in backup.sh)

3. **Test restore procedures** regularly

### Volume Security

1. **Use non-root user** in containers
2. **Set appropriate permissions** on mounted directories
3. **Encrypt sensitive data** at rest
4. **Limit volume access** to required services only

### Performance Optimization

1. **Use local driver** for volumes
2. **Place volumes on SSD** for better performance
3. **Monitor volume sizes** to prevent disk exhaustion
4. **Implement log rotation** to manage log volume size

## Troubleshooting

### Common Issues

#### Permission Denied

```bash
# Fix ownership
docker exec cosmos-api chown -R appuser:appuser /app/data

# Fix permissions
docker exec cosmos-api chmod -R 755 /app/data
```

#### Volume Not Found

```bash
# Create missing volume
docker volume create cosmos-coherence_postgres-data

# Recreate all volumes
docker compose up -d
```

#### Disk Space Issues

```bash
# Check disk usage
df -h

# Clean Docker system
docker system prune -a --volumes

# Remove old backups
find ./backups -name "*.tar.gz" -mtime +30 -delete
```

### Recovery Procedures

#### Corrupted Database

1. Stop services: `docker compose down`
2. Restore from last known good backup
3. Verify data integrity
4. Start services: `docker compose up -d`

#### Lost Volume

1. Check if volume exists: `docker volume ls`
2. If missing, restore from backup
3. If no backup, recreate and reinitialize

## Monitoring

### Health Checks

```bash
# Run health check script
./scripts/health-check.sh

# Check specific service
docker compose ps db
docker compose logs db --tail=50
```

### Volume Metrics

```bash
# Monitor volume usage
watch -n 60 'docker system df -v | grep cosmos-coherence'

# Alert on high usage (example)
USAGE=$(docker run --rm -v cosmos-coherence_postgres-data:/data alpine df /data | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$USAGE" -gt 80 ]; then
    echo "WARNING: Volume usage above 80%"
fi
```

## Migration Guide

### Migrating to New Host

1. **On source host**:
   ```bash
   ./scripts/backup.sh
   scp backups/latest.tar.gz user@newhost:/path/
   ```

2. **On target host**:
   ```bash
   ./scripts/restore.sh latest.tar.gz
   ```

### Upgrading PostgreSQL

1. Backup current data
2. Export with `pg_dumpall`
3. Stop old container
4. Start new version container
5. Import with `psql`

## References

- [Docker Volumes Documentation](https://docs.docker.com/storage/volumes/)
- [PostgreSQL Backup/Restore](https://www.postgresql.org/docs/current/backup.html)
- [Redis Persistence](https://redis.io/topics/persistence)
- [Docker Compose Volumes](https://docs.docker.com/compose/compose-file/compose-file-v3/#volumes)
