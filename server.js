// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// 🔥 TIMEOUT / RETRY CONFIG
// Render's edge proxy has a hard ~100s limit on most plans, so we stay under it.
const NIM_REQUEST_TIMEOUT_MS = parseInt(process.env.NIM_REQUEST_TIMEOUT_MS || '90000', 10);
const NIM_MAX_RETRIES = parseInt(process.env.NIM_MAX_RETRIES || '2', 10);
const RETRYABLE_STATUS_CODES = [502, 503, 504];

// Default max_tokens — lowered from 9024 to reduce odds of hitting upstream
// timeouts on non-streaming requests. Override per-request or via env var.
const DEFAULT_MAX_TOKENS = parseInt(process.env.DEFAULT_MAX_TOKENS || '4096', 10);

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'deepseek-ai/deepseek-v4-pro',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v4-flash',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Calls the NIM chat/completions endpoint with a timeout and automatic
 * retry-with-backoff on transient upstream failures (502/503/504).
 * Does NOT retry once a stream has already started — only retries the
 * initial request/connection attempt.
 */
async function callNimWithRetry(nimRequest, headers) {
  let lastError;

  for (let attempt = 0; attempt <= NIM_MAX_RETRIES; attempt++) {
    try {
      return await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
        headers,
        responseType: nimRequest.stream ? 'stream' : 'json',
        timeout: NIM_REQUEST_TIMEOUT_MS
      });
    } catch (err) {
      lastError = err;
      const status = err.response?.status;
      const isRetryable = RETRYABLE_STATUS_CODES.includes(status);

      if (isRetryable && attempt < NIM_MAX_RETRIES) {
        const backoffMs = 1000 * (attempt + 1);
        console.warn(
          `NIM request failed with status ${status} (attempt ${attempt + 1}/${NIM_MAX_RETRIES + 1}). ` +
          `Retrying in ${backoffMs}ms...`
        );
        await sleep(backoffMs);
        continue;
      }

      throw err;
    }
  }

  throw lastError;
}

/**
 * Tries to resolve a model name not present in MODEL_MAPPING by checking
 * whether NIM accepts it directly, falling back to a size-based guess.
 */
async function resolveUnmappedModel(model, headers) {
  try {
    const probe = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      { model, messages: [{ role: 'user', content: 'test' }], max_tokens: 1 },
      { headers, timeout: 10000, validateStatus: (status) => status < 500 }
    );

    if (probe.status >= 200 && probe.status < 300) {
      return model;
    }
  } catch (e) {
    // Probe failed (network error, timeout, etc.) — fall through to heuristic guess.
  }

  const modelLower = model.toLowerCase();
  if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
    return 'meta/llama-3.1-405b-instruct';
  }
  if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
    return 'meta/llama-3.1-70b-instruct';
  }
  return 'meta/llama-3.1-8b-instruct';
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    request_timeout_ms: NIM_REQUEST_TIMEOUT_MS,
    max_retries: NIM_MAX_RETRIES
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((model) => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    const headers = {
      Authorization: `Bearer ${NIM_API_KEY}`,
      'Content-Type': 'application/json'
    };

    // Resolve model: use mapping, or probe/guess if unmapped
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      nimModel = await resolveUnmappedModel(model, headers);
    }

    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || DEFAULT_MAX_TOKENS,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };

    // Make request to NVIDIA NIM API (with timeout + retry on transient failures)
    const response = await callNimWithRetry(nimRequest, headers);

    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach((line) => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }

            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;

                if (SHOW_REASONING) {
                  let combinedContent = '';

                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }

                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }

                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map((choice) => {
          let fullContent = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }

          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }
  } catch (error) {
    const status = error.response?.status || 500;
    console.error(`Proxy error (status ${status}):`, error.message);

    res.status(status).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: status
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Request timeout: ${NIM_REQUEST_TIMEOUT_MS}ms | Max retries: ${NIM_MAX_RETRIES}`);
});
