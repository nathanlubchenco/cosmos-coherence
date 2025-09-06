#!/bin/bash
# Health check script for Cosmos Coherence services

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXIT_CODE=0

# Health check functions
check_service() {
    local service_name=$1
    local check_command=$2

    printf "Checking %-15s ... " "$service_name"

    if eval "$check_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Healthy${NC}"
        return 0
    else
        echo -e "${RED}✗ Unhealthy${NC}"
        EXIT_CODE=1
        return 1
    fi
}

check_docker_service() {
    local service_name=$1
    local container_name=$2

    printf "Checking %-15s ... " "$service_name"

    if docker ps | grep -q "$container_name"; then
        # Check if container is running
        if [ "$(docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null)" == "true" ]; then
            # Check if container is healthy (if health check is defined)
            local health_status=$(docker inspect -f '{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "none")

            if [ "$health_status" == "healthy" ] || [ "$health_status" == "none" ]; then
                echo -e "${GREEN}✓ Running${NC}"
                return 0
            else
                echo -e "${YELLOW}⚠ Running (health: $health_status)${NC}"
                return 0
            fi
        else
            echo -e "${RED}✗ Not running${NC}"
            EXIT_CODE=1
            return 1
        fi
    else
        echo -e "${RED}✗ Container not found${NC}"
        EXIT_CODE=1
        return 1
    fi
}

# Main health checks
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}   Cosmos Coherence Health Check${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo ""

# Check Docker services
echo "Docker Services:"
echo "----------------"
check_docker_service "API" "cosmos-api"
check_docker_service "Dashboard" "cosmos-dashboard"
check_docker_service "Database" "cosmos-db"
check_docker_service "Redis" "cosmos-redis"

# Check if development services are running
if docker ps | grep -q "cosmos-adminer"; then
    check_docker_service "Adminer" "cosmos-adminer"
fi

if docker ps | grep -q "cosmos-mailhog"; then
    check_docker_service "MailHog" "cosmos-mailhog"
fi

echo ""

# Check HTTP endpoints
echo "HTTP Endpoints:"
echo "---------------"
check_service "API Health" "curl -f http://localhost:8000/health"
check_service "API Root" "curl -f http://localhost:8000/"
check_service "Dashboard" "curl -f http://localhost:8050/"

echo ""

# Check database connectivity
echo "Database:"
echo "---------"
check_service "PostgreSQL" "docker exec cosmos-db pg_isready -U cosmos -d cosmos_coherence"

echo ""

# Check Redis connectivity
echo "Cache:"
echo "------"
check_service "Redis" "docker exec cosmos-redis redis-cli ping"

echo ""

# Check disk space
echo "System Resources:"
echo "-----------------"
printf "Checking %-15s ... " "Disk Space"
DISK_USAGE=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 80 ]; then
    echo -e "${GREEN}✓ ${DISK_USAGE}% used${NC}"
else
    echo -e "${YELLOW}⚠ ${DISK_USAGE}% used${NC}"
fi

# Check Docker resources
printf "Checking %-15s ... " "Docker"
if docker system df > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Available${NC}"
else
    echo -e "${RED}✗ Not available${NC}"
    EXIT_CODE=1
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════${NC}"

# Summary
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All health checks passed!${NC}"
else
    echo -e "${RED}✗ Some health checks failed${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "- Check logs: docker compose logs"
    echo "- Restart services: docker compose restart"
    echo "- Check configuration: cat .env"
fi

exit $EXIT_CODE
