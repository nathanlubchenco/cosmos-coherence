#!/bin/bash
# Start production environment for Cosmos Coherence

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}Starting Cosmos Coherence Production Environment${NC}"

cd "$PROJECT_ROOT"

# Check for production environment file
if [ ! -f .env.production ]; then
    echo -e "${RED}Error: .env.production file not found${NC}"
    echo "Please create .env.production with production settings"
    exit 1
fi

# Confirm production deployment
echo -e "${YELLOW}⚠️  WARNING: Starting PRODUCTION environment${NC}"
read -p "Are you sure you want to continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Use production environment
cp .env.production .env

# Build production images
echo "Building production containers..."
docker compose -f docker-compose.yml -f docker-compose.prod.yml build

# Start production stack
echo "Starting production services..."
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo "Waiting for services to be healthy..."
sleep 10

# Run health checks
"$SCRIPT_DIR/health-check.sh"

# Check service status
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps

# Display status
echo -e "\n${GREEN}Production environment is running!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 API:       http://localhost:8000"
echo "📊 Dashboard: http://localhost:8050"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "View logs:    docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f"
echo "Stop:         docker compose -f docker-compose.yml -f docker-compose.prod.yml down"
