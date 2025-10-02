
var builder = WebApplication.CreateBuilder(args);

builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowFrontend", policy =>
        policy.WithOrigins("http://localhost:3000").AllowAnyHeader().AllowAnyMethod());
});

var mlBase = Environment.GetEnvironmentVariable("ML_BASE_URL") ?? "http://localhost:8000";
builder.Services.AddHttpClient("ml", c => c.BaseAddress = new Uri(mlBase));

builder.Services.AddControllers();
var app = builder.Build();

app.UseCors("AllowFrontend");
app.MapControllers();

app.Run();
