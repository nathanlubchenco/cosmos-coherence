#!/bin/bash
# Restore script for Cosmos Coherence

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_ROOT="${BACKUP_ROOT:-$PROJECT_ROOT/backups}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# List available backups
list_backups() {
    echo "Available backups:"
    echo "────────────────────────────────────"

    if [ -d "$BACKUP_ROOT" ]; then
        for backup in "$BACKUP_ROOT"/*.tar.gz; do
            if [ -f "$backup" ]; then
                filename=$(basename "$backup")
                size=$(du -h "$backup" | cut -f1)
                date=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$backup" 2>/dev/null || stat -c "%y" "$backup" 2>/dev/null | cut -d' ' -f1,2)
                echo "  • $filename ($size) - $date"
            fi
        done
    else
        echo "  No backups found in $BACKUP_ROOT"
        exit 1
    fi

    echo "────────────────────────────────────"
}

# Select backup to restore
select_backup() {
    if [ -z "$1" ]; then
        list_backups
        echo ""
        read -p "Enter backup filename (e.g., 20240101_120000.tar.gz): " BACKUP_FILE
    else
        BACKUP_FILE="$1"
    fi

    BACKUP_PATH="$BACKUP_ROOT/$BACKUP_FILE"

    if [ ! -f "$BACKUP_PATH" ]; then
        log_error "Backup file not found: $BACKUP_PATH"
        exit 1
    fi

    log_info "Selected backup: $BACKUP_FILE"
}

# Extract backup
extract_backup() {
    log_info "Extracting backup..."

    TEMP_DIR=$(mktemp -d)
    tar xzf "$BACKUP_PATH" -C "$TEMP_DIR"

    # Find the extracted directory
    BACKUP_DIR=$(find "$TEMP_DIR" -maxdepth 1 -type d | grep -v "^$TEMP_DIR$" | head -1)

    if [ ! -d "$BACKUP_DIR" ]; then
        log_error "Failed to extract backup"
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    echo "$BACKUP_DIR"
}

# Stop services
stop_services() {
    log_info "Stopping services..."

    cd "$PROJECT_ROOT"
    docker compose down || true

    log_info "Services stopped"
}

# Restore database
restore_database() {
    local backup_dir=$1

    if [ -f "$backup_dir/database.sql.gz" ]; then
        log_info "Restoring database..."

        # Start only database service
        docker compose up -d db

        # Wait for database to be ready
        log_info "Waiting for database to be ready..."
        sleep 10

        # Restore database
        gunzip < "$backup_dir/database.sql.gz" | docker exec -i cosmos-db psql -U cosmos cosmos_coherence

        log_info "Database restored"
    else
        log_warn "No database backup found, skipping database restore"
    fi
}

# Restore Redis
restore_redis() {
    local backup_dir=$1

    if [ -f "$backup_dir/redis_dump.rdb" ]; then
        log_info "Restoring Redis data..."

        # Start Redis service
        docker compose up -d redis

        # Stop Redis to restore data
        docker compose stop redis

        # Copy dump file
        docker cp "$backup_dir/redis_dump.rdb" cosmos-redis:/data/dump.rdb

        # Start Redis again
        docker compose up -d redis

        log_info "Redis data restored"
    else
        log_warn "No Redis backup found, skipping Redis restore"
    fi
}

# Restore Docker volumes
restore_volumes() {
    local backup_dir=$1

    log_info "Restoring Docker volumes..."

    # List of volumes to restore
    VOLUMES=(
        "cosmos-coherence_postgres-data"
        "cosmos-coherence_redis-data"
        "cosmos-coherence_api-cache"
        "cosmos-coherence_dashboard-cache"
    )

    for volume in "${VOLUMES[@]}"; do
        if [ -f "$backup_dir/${volume}.tar.gz" ]; then
            log_info "Restoring volume: $volume"

            # Create volume if it doesn't exist
            docker volume create "$volume" 2>/dev/null || true

            # Restore volume data
            docker run --rm \
                -v "$volume:/target" \
                -v "$backup_dir:/backup:ro" \
                alpine sh -c "cd /target && tar xzf /backup/${volume}.tar.gz"
        fi
    done

    log_info "Volumes restored"
}

# Restore configuration files
restore_configs() {
    local backup_dir=$1

    log_info "Restoring configuration files..."

    # Restore environment files
    if [ -f "$backup_dir/configs/.env" ]; then
        log_warn "Found .env in backup. Review and restore manually if needed:"
        echo "  cp $backup_dir/configs/.env $PROJECT_ROOT/.env"
    fi

    if [ -f "$backup_dir/configs/.env.production" ]; then
        log_warn "Found .env.production in backup. Review and restore manually if needed:"
        echo "  cp $backup_dir/configs/.env.production $PROJECT_ROOT/.env.production"
    fi

    # Restore configs directory
    if [ -d "$backup_dir/configs" ]; then
        log_info "Configuration files found in backup"
        log_warn "Review configuration files in: $backup_dir/configs"
    fi

    log_info "Configuration review complete"
}

# Display backup info
show_backup_info() {
    local backup_dir=$1

    if [ -f "$backup_dir/backup_info.txt" ]; then
        echo ""
        echo "═══════════════════════════════════════"
        echo "   Backup Information"
        echo "═══════════════════════════════════════"
        cat "$backup_dir/backup_info.txt"
        echo "═══════════════════════════════════════"
        echo ""
    fi
}

# Start services
start_services() {
    log_info "Starting all services..."

    cd "$PROJECT_ROOT"
    docker compose up -d

    # Wait for services to be healthy
    sleep 10

    # Run health check
    if [ -f "$SCRIPT_DIR/health-check.sh" ]; then
        "$SCRIPT_DIR/health-check.sh"
    fi

    log_info "Services started"
}

# Main execution
main() {
    echo -e "${GREEN}═══════════════════════════════════════${NC}"
    echo -e "${GREEN}   Cosmos Coherence Restore Utility${NC}"
    echo -e "${GREEN}═══════════════════════════════════════${NC}"
    echo ""

    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running"
        exit 1
    fi

    # Select backup
    select_backup "$1"

    # Confirm restoration
    echo ""
    log_warn "This will restore data from: $BACKUP_FILE"
    log_warn "Current data will be OVERWRITTEN!"
    echo ""
    read -p "Are you sure you want to continue? (y/N) " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Restore cancelled"
        exit 0
    fi

    # Extract backup
    BACKUP_DIR=$(extract_backup)

    # Show backup info
    show_backup_info "$BACKUP_DIR"

    # Stop services
    stop_services

    # Restore components
    restore_volumes "$BACKUP_DIR"
    restore_database "$BACKUP_DIR"
    restore_redis "$BACKUP_DIR"
    restore_configs "$BACKUP_DIR"

    # Start services
    start_services

    # Cleanup
    rm -rf "$(dirname "$BACKUP_DIR")"

    echo ""
    log_info "✓ Restore completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Verify services are running: docker compose ps"
    echo "2. Check application functionality"
    echo "3. Review any configuration files that need manual restoration"
}

# Handle errors
trap 'log_error "Restore failed!"; exit 1' ERR

# Run main function
main "$@"
