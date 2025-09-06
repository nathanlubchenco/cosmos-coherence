#!/bin/bash
# Deployment script for Cosmos Coherence

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-production}"

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

check_requirements() {
    log_info "Checking requirements..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi

    # Check environment file
    if [ "$ENVIRONMENT" == "production" ]; then
        if [ ! -f "$PROJECT_ROOT/.env.production" ]; then
            log_error ".env.production file not found"
            exit 1
        fi
    fi

    log_info "All requirements satisfied"
}

backup_data() {
    log_info "Creating backup..."

    BACKUP_DIR="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"

    # Backup database if running
    if docker ps | grep -q cosmos-db; then
        log_info "Backing up database..."
        docker exec cosmos-db pg_dump -U cosmos cosmos_coherence | gzip > "$BACKUP_DIR/database.sql.gz"
    fi

    # Backup volumes
    log_info "Backing up Docker volumes..."
    docker run --rm \
        -v cosmos-coherence_postgres-data:/data/postgres \
        -v cosmos-coherence_redis-data:/data/redis \
        -v "$BACKUP_DIR:/backup" \
        alpine tar czf /backup/volumes.tar.gz /data

    log_info "Backup completed: $BACKUP_DIR"
}

deploy_production() {
    log_info "Deploying to production..."

    cd "$PROJECT_ROOT"

    # Load production environment
    cp .env.production .env

    # Pull latest images
    log_info "Pulling latest images..."
    docker compose -f docker-compose.yml -f docker-compose.prod.yml pull

    # Build production images
    log_info "Building production images..."
    docker compose -f docker-compose.yml -f docker-compose.prod.yml build --no-cache

    # Stop existing containers
    log_info "Stopping existing containers..."
    docker compose down

    # Start production stack
    log_info "Starting production stack..."
    docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

    # Wait for health checks
    log_info "Waiting for services to be healthy..."
    sleep 10

    # Run health checks
    "$SCRIPT_DIR/health-check.sh"

    log_info "Production deployment completed successfully!"
}

deploy_development() {
    log_info "Deploying development environment..."

    cd "$PROJECT_ROOT"

    # Load development environment
    if [ -f .env.development ]; then
        cp .env.development .env
    fi

    # Build development images
    log_info "Building development images..."
    docker compose build

    # Start development stack
    log_info "Starting development stack..."
    docker compose up -d

    # Wait for services
    sleep 10

    # Run health checks
    "$SCRIPT_DIR/health-check.sh"

    log_info "Development deployment completed!"
    log_info "API: http://localhost:8000"
    log_info "Dashboard: http://localhost:8050"
    log_info "Adminer: http://localhost:8080"
    log_info "MailHog: http://localhost:8025"
}

# Main execution
main() {
    log_info "Starting deployment for environment: $ENVIRONMENT"

    check_requirements

    if [ "$ENVIRONMENT" == "production" ]; then
        log_warn "Deploying to PRODUCTION environment"
        read -p "Are you sure you want to continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled"
            exit 0
        fi

        backup_data
        deploy_production
    else
        deploy_development
    fi
}

# Run main function
main "$@"
