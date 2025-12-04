#!/bin/bash
# Install Redis (for dev containers without systemd)

echo "Installing Redis..."
sudo apt update && sudo apt install -y redis-server

# Start Redis using service command (for containers)
echo "Starting Redis service..."
sudo service redis-server start

# Wait a moment for Redis to start
sleep 2

# Test Redis connection
echo "Testing Redis connection..."
python -c "import redis; r = redis.Redis(host='localhost', port=6379, db=0); print('✅ Redis Status:', 'Connected' if r.ping() else 'Failed'); print('✅ Redis Info:', r.info('server')['redis_version'])"

echo ""
echo "============================================"
echo "✅ REDIS SETUP COMPLETE!"
echo "============================================"
echo "Redis is running on: localhost:6379"
echo "To check status: sudo service redis-server status"
echo "To stop: sudo service redis-server stop"
echo "To restart: sudo service redis-server restart"
echo "============================================"