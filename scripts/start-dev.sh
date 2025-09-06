#!/bin/bash
# Start development environment for Cosmos Coherence

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}Starting Cosmos Coherence Development Environment${NC}"

cd "$PROJECT_ROOT"

# Check for .env file
if [ ! -f .env ]; then
    if [ -f .env.development ]; then
        echo "Copying .env.development to .env..."
        cp .env.development .env
    elif [ -f .env.example ]; then
        echo -e "${YELLOW}Warning: No .env file found. Copying from .env.example${NC}"
        cp .env.example .env
        echo "Please update .env with your API keys and settings"
    fi
fi

# Build and start containers
echo "Building development containers..."
docker compose build

echo "Starting services..."
docker compose up -d

# Wait for services to be ready
echo "Waiting for services to be healthy..."
sleep 5

# Check service health
docker compose ps

# Show logs for any failed services
if docker compose ps | grep -q "Exit"; then
    echo -e "${YELLOW}Some services failed to start. Showing logs:${NC}"
    docker compose logs --tail=50
fi

# Display access URLs
echo -e "\n${GREEN}Development environment is running!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 API:       http://localhost:8000"
echo "📊 Dashboard: http://localhost:8050"
echo "🗄️  Database:  http://localhost:8080 (Adminer)"
echo "📧 MailHog:   http://localhost:8025"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "View logs:    docker compose logs -f"
echo "Stop:         docker compose down"
echo "Clean stop:   docker compose down -v"
