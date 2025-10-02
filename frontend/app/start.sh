#!/bin/sh
set -e

# Ensure CRA binds to all interfaces inside the container
export HOST=0.0.0.0
export PORT=3000

npm start

