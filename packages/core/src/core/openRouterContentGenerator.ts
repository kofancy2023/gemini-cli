/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  GenerateContentParameters,
  GenerateContentResponse,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  Part,
  Content,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
import type { Config } from '../config/config.js';

export class OpenRouterContentGenerator implements ContentGenerator {
  private apiKey: string;
  private baseUrl: string;
  private model: string;

  constructor(apiKey: string, _config: Config) {
    this.apiKey = apiKey;
    // 允许用户自定义 Base URL，默认为 OpenRouter
    this.baseUrl =
      process.env['OPENROUTER_BASE_URL'] || 'https://openrouter.ai/api/v1';
    this.model =
      process.env['OPENROUTER_MODEL'] || 'google/gemini-2.0-flash-001';
  }

  async generateContent(
    request: GenerateContentParameters,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    const messages = this.convertContentsToMessages(request.contents);

    // 从 request.config 中提取配置
    const config = request.config as
      | {
          temperature?: number;
          topP?: number;
          maxOutputTokens?: number;
        }
      | undefined;

    const body = {
      model: this.model,
      messages: messages,
      temperature: config?.temperature ?? 0.7,
      top_p: config?.topP ?? 1.0,
      max_tokens: config?.maxOutputTokens,
    };

    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
        'HTTP-Referer': 'https://github.com/google/gemini-cli',
        'X-Title': 'Gemini CLI',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `OpenRouter API Error: ${response.status} ${response.statusText} - ${errorText}`,
      );
    }

    const data = await response.json();
    return this.convertOpenAIResponseToGemini(data);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const messages = this.convertContentsToMessages(request.contents);

    // 从 request.config 中提取配置
    const config = request.config as
      | {
          temperature?: number;
          topP?: number;
          maxOutputTokens?: number;
        }
      | undefined;

    const body = {
      model: this.model,
      messages: messages,
      stream: true,
      temperature: config?.temperature ?? 0.7,
      top_p: config?.topP ?? 1.0,
      max_tokens: config?.maxOutputTokens,
    };

    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
        'HTTP-Referer': 'https://github.com/google/gemini-cli',
        'X-Title': 'Gemini CLI',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `OpenRouter API Error: ${response.status} ${response.statusText} - ${errorText}`,
      );
    }

    if (!response.body) throw new Error('No response body');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    return (async function* () {
      let buffer = '';
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            if (trimmed === 'data: [DONE]') return; // Explicitly end the generator

            if (trimmed.startsWith('data: ')) {
              try {
                const data = JSON.parse(trimmed.slice(6));
                const content = data.choices[0]?.delta?.content;
                if (content) {
                  yield {
                    candidates: [
                      {
                        content: {
                          parts: [{ text: content }],
                          role: 'model',
                        },
                        finishReason:
                          data.choices[0]?.finish_reason || undefined,
                      },
                    ],
                    text: content,
                  } as GenerateContentResponse;
                }
              } catch (e) {
                // Ignore parse errors for partial lines
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    })();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // 简易实现：估算 token 数
    let text = '';
    if (typeof request.contents === 'string') {
      text = request.contents;
    } else if (Array.isArray(request.contents)) {
      text = request.contents
        .map((c: Content) => {
          const parts = c.parts || [];
          return parts
            .map((p) => ('text' in p ? p.text : ''))
            .filter(Boolean)
            .join('');
        })
        .join('');
    }

    return { totalTokens: Math.ceil(text.length / 4) };
  }

  async embedContent(
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error('Embed content not supported via OpenRouter yet.');
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private convertContentsToMessages(contents: any): any[] {
    if (typeof contents === 'string') {
      return [{ role: 'user', content: contents }];
    }

    if (!Array.isArray(contents)) {
      return [];
    }

    return contents.map((content: Content) => {
      let role = content.role === 'model' ? 'assistant' : content.role;
      if (!role) role = 'user'; // default to user

      const parts = content.parts || [];
      const textParts = parts
        .filter((p: Part) => 'text' in p && p.text)
        .map((p: Part) => ('text' in p ? p.text : ''))
        .join('');

      return {
        role: role,
        content: textParts,
      };
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private convertOpenAIResponseToGemini(data: any): GenerateContentResponse {
    const content = data.choices[0]?.message?.content || '';

    // 创建一个完整的 GenerateContentResponse 对象
    const response: GenerateContentResponse = {
      candidates: [
        {
          content: {
            parts: [{ text: content }],
            role: 'model',
          },
          finishReason: data.choices[0]?.finish_reason,
          index: 0,
        },
      ],
      text: content,
      functionCalls: () => [],
      executableCode: undefined,
      codeExecutionResult: undefined,
      data: data,
    } as unknown as GenerateContentResponse;

    return response;
  }
}
