/**
 * WASM TTS benchmark — runs in headless Chromium via Playwright.
 *
 * Usage: bunx playwright test tests/bench_tts_wasm.spec.ts
 */
import { test, expect } from '@playwright/test';

test('Q4 TTS WASM benchmark', async ({ browser }) => {
    const context = await browser.newContext({ ignoreHTTPSErrors: true });
    const page = await context.newPage();
    // Collect console output
    const logs: string[] = [];
    page.on('console', msg => {
        const text = msg.text();
        logs.push(text);
        console.log(`[browser] ${text}`);
    });

    await page.goto('https://localhost:8443/bench-tts.html');

    // Wait for benchmark to complete (up to 10 minutes for shard download + inference)
    await page.waitForFunction(() => (window as any).__benchmarkDone === true, {
        timeout: 600_000,
    });

    // Check for errors
    const error = await page.evaluate(() => (window as any).__benchmarkError);
    expect(error).toBeUndefined();

    // Print full log
    console.log('\n=== WASM TTS Benchmark Results ===');
    for (const line of logs) {
        console.log(line);
    }

    // Verify we got results
    expect(logs.some(l => l.includes('BENCHMARK COMPLETE'))).toBe(true);
});
