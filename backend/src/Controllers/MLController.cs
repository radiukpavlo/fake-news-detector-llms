
using Microsoft.AspNetCore.Mvc;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading;
using System.Threading.Tasks;

namespace RealFakeNews.Controllers;

[ApiController]
[Route("api/ml")]
public class MLController : ControllerBase
{
    private readonly HttpClient _http;
    private readonly string _mlBaseUrl;

    public MLController(IHttpClientFactory factory)
    {
        _http = factory.CreateClient("ml");
        _mlBaseUrl = _http.BaseAddress?.ToString()?.TrimEnd('/') ?? "http://localhost:8000";
    }

    private async Task<IActionResult> ForwardResponse(HttpResponseMessage response)
    {
        var contentType = response.Content.Headers.ContentType?.MediaType ?? "application/json";
        var payload = await response.Content.ReadAsStringAsync();

        return new ContentResult
        {
            Content = payload,
            ContentType = contentType,
            StatusCode = (int)response.StatusCode,
        };
    }

    [HttpGet("health")]
    public Task<IActionResult> Health(CancellationToken ct) => ProxyGet("health", ct);

    [HttpGet("models")]
    public Task<IActionResult> Models(CancellationToken ct) => ProxyGet("models", ct);

    [HttpPost("train")]
    public Task<IActionResult> Train([FromBody] object body, CancellationToken ct) => ProxyPost("train", body, ct);

    [HttpGet("train/status/{modelName}")]
    public Task<IActionResult> TrainStatus(string modelName, CancellationToken ct) => ProxyGet($"train/status/{modelName}", ct);

    [HttpPost("predict")]
    public Task<IActionResult> Predict([FromBody] object body, CancellationToken ct) => ProxyPost("predict", body, ct);

    [HttpGet("metrics/{modelName}")]
    public Task<IActionResult> Metrics(string modelName, CancellationToken ct) => ProxyGet($"metrics/{modelName}", ct);

    [HttpGet("metrics/plots/{modelName}/{plotName}")]
    public Task<IActionResult> Plot(string modelName, string plotName, CancellationToken ct) => ProxyGet($"metrics/plots/{modelName}/{plotName}", ct);

    [HttpPost("explain")]
    public Task<IActionResult> Explain([FromBody] object body, CancellationToken ct) => ProxyPost("explain", body, ct);

    [HttpPost("project")]
    public Task<IActionResult> Project([FromBody] object body, CancellationToken ct) => ProxyPost("project", body, ct);

    [HttpGet("report/download/{modelName}")]
    public Task<IActionResult> Report(string modelName, CancellationToken ct) => ProxyGet($"report/download/{modelName}", ct);

    private async Task<IActionResult> ProxyGet(string path, CancellationToken ct)
    {
        var resp = await _http.GetAsync($"{_mlBaseUrl}/{path}", ct);
        return await ForwardResponse(resp);
    }
    private async Task<IActionResult> ProxyPost(string path, object body, CancellationToken ct)
    {
        var resp = await _http.PostAsJsonAsync($"{_mlBaseUrl}/{path}", body, ct);
        return await ForwardResponse(resp);
    }
}
