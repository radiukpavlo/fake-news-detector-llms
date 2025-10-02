FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy and restore the project
COPY src/RealFakeNews.csproj ./src/
RUN dotnet restore ./src/RealFakeNews.csproj

# Copy the source and publish
COPY src/ ./src/
RUN dotnet publish ./src/RealFakeNews.csproj -c Release -o /app/publish

FROM mcr.microsoft.com/dotnet/aspnet:9.0
WORKDIR /app
COPY --from=build /app/publish .

# Healthcheck in docker-compose.yml uses curl
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV ASPNETCORE_URLS=http://+:5000
EXPOSE 5000
ENTRYPOINT ["dotnet", "RealFakeNews.dll"]
