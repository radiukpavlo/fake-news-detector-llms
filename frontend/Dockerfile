# Base Node.js image
FROM node:22-alpine

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json from the app subdirectory
# The build context is `frontend/`, so we need to specify `app/`
COPY app/package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code from the app subdirectory
COPY app/ .

# Make start script executable
RUN chmod +x start.sh

# Expose port for React app
EXPOSE 3000

# Start command
CMD ["./start.sh"]