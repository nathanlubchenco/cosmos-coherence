#!/bin/bash
# Backup script for Cosmos Coherence

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_ROOT="${BACKUP_ROOT:-$PROJECT_ROOT/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/$TIMESTAMP"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo -e "${GREEN}Starting backup to: $BACKUP_DIR${NC}"

# Backup database
backup_database() {
    echo "Backing up PostgreSQL database..."

    if docker ps | grep -q cosmos-db; then
        docker exec cosmos-db pg_dump -U cosmos cosmos_coherence | gzip > "$BACKUP_DIR/database.sql.gz"
        echo "Database backed up successfully"
    else
        echo -e "${YELLOW}Warning: Database container not running, skipping database backup${NC}"
    fi
}

# Backup Redis
backup_redis() {
    echo "Backing up Redis data..."

    if docker ps | grep -q cosmos-redis; then
        docker exec cosmos-redis redis-cli BGSAVE
        sleep 2
        docker cp cosmos-redis:/data/dump.rdb "$BACKUP_DIR/redis_dump.rdb"
        echo "Redis backed up successfully"
    else
        echo -e "${YELLOW}Warning: Redis container not running, skipping Redis backup${NC}"
    fi
}

# Backup Docker volumes
backup_volumes() {
    echo "Backing up Docker volumes..."

    # List of volumes to backup
    VOLUMES=(
        "cosmos-coherence_postgres-data"
        "cosmos-coherence_redis-data"
        "cosmos-coherence_api-cache"
        "cosmos-coherence_dashboard-cache"
    )

    for volume in "${VOLUMES[@]}"; do
        if docker volume ls | grep -q "$volume"; then
            echo "Backing up volume: $volume"
            docker run --rm \
                -v "$volume:/source:ro" \
                -v "$BACKUP_DIR:/backup" \
                alpine tar czf "/backup/${volume}.tar.gz" -C /source .
        fi
    done

    echo "Volumes backed up successfully"
}

# Backup configuration files
backup_configs() {
    echo "Backing up configuration files..."

    # Create config backup directory
    mkdir -p "$BACKUP_DIR/configs"

    # Copy configuration files
    if [ -f "$PROJECT_ROOT/.env" ]; then
        cp "$PROJECT_ROOT/.env" "$BACKUP_DIR/configs/"
    fi

    if [ -f "$PROJECT_ROOT/.env.production" ]; then
        cp "$PROJECT_ROOT/.env.production" "$BACKUP_DIR/configs/"
    fi

    if [ -d "$PROJECT_ROOT/configs" ]; then
        cp -r "$PROJECT_ROOT/configs" "$BACKUP_DIR/"
    fi

    # Copy docker compose files
    cp "$PROJECT_ROOT/docker-compose.yml" "$BACKUP_DIR/configs/" 2>/dev/null || true
    cp "$PROJECT_ROOT/docker-compose.prod.yml" "$BACKUP_DIR/configs/" 2>/dev/null || true
    cp "$PROJECT_ROOT/docker-compose.override.yml" "$BACKUP_DIR/configs/" 2>/dev/null || true

    echo "Configuration files backed up successfully"
}

# Create backup metadata
create_metadata() {
    echo "Creating backup metadata..."

    cat > "$BACKUP_DIR/backup_info.txt" << EOF
Cosmos Coherence Backup
========================
Date: $(date)
Timestamp: $TIMESTAMP
Host: $(hostname)
User: $(whoami)

Docker Status:
$(docker compose ps 2>/dev/null || echo "Not available")

Disk Usage:
$(df -h "$PROJECT_ROOT" | tail -1)

Backup Contents:
$(ls -la "$BACKUP_DIR")
EOF

    echo "Metadata created"
}

# Compress entire backup
compress_backup() {
    echo "Compressing backup..."

    cd "$BACKUP_ROOT"
    tar czf "${TIMESTAMP}.tar.gz" "$TIMESTAMP"

    # Remove uncompressed backup directory
    rm -rf "$TIMESTAMP"

    echo "Backup compressed: ${TIMESTAMP}.tar.gz"
}

# Clean old backups (keep last 7 days)
clean_old_backups() {
    echo "Cleaning old backups..."

    # Find and delete backups older than 7 days
    find "$BACKUP_ROOT" -name "*.tar.gz" -mtime +7 -delete

    echo "Old backups cleaned"
}

# Main execution
main() {
    echo -e "${GREEN}═══════════════════════════════════════${NC}"
    echo -e "${GREEN}   Cosmos Coherence Backup Utility${NC}"
    echo -e "${GREEN}═══════════════════════════════════════${NC}"

    # Run backup tasks
    backup_database
    backup_redis
    backup_volumes
    backup_configs
    create_metadata
    compress_backup
    clean_old_backups

    # Calculate backup size
    BACKUP_SIZE=$(du -h "$BACKUP_ROOT/${TIMESTAMP}.tar.gz" | cut -f1)

    echo -e "\n${GREEN}✓ Backup completed successfully!${NC}"
    echo "Location: $BACKUP_ROOT/${TIMESTAMP}.tar.gz"
    echo "Size: $BACKUP_SIZE"
}

# Handle errors
trap 'echo -e "${RED}Backup failed!${NC}"; exit 1' ERR

# Run main function
main "$@"
