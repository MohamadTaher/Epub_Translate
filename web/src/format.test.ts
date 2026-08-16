import { describe, expect, it } from 'vitest';
import { clockTime, duration, estimateSeconds, fileSize, languageLabel, tokens } from '@/format';

describe('clockTime', () => {
  it('is blank when the event carried no timestamp', () => {
    expect(clockTime(undefined)).toBe('');
  });

  it('is a fixed-width 24-hour clock, whatever the reader\'s locale', () => {
    // Not asserting the hour: the value is rendered in the reader's timezone.
    expect(clockTime(Date.UTC(2024, 0, 1, 14, 3, 22) / 1000)).toMatch(/^\d{2}:\d{2}:\d{2}$/);
  });
});

describe('tokens', () => {
  it('counts under a thousand exactly', () => {
    expect(tokens(0)).toBe('0');
    expect(tokens(999)).toBe('999');
  });

  it('gains a decimal from a thousand, and loses it again at ten', () => {
    expect(tokens(1000)).toBe('1.0k');
    expect(tokens(9999)).toBe('10.0k');
    expect(tokens(10_000)).toBe('10k');
    expect(tokens(10_499)).toBe('10k');
  });
});

describe('fileSize', () => {
  it('reports KB below 0.1 MB and MB from there up', () => {
    // 0.1 MB is 104857.6 bytes, so these two straddle the boundary.
    expect(fileSize(104_857)).toBe('102 KB');
    expect(fileSize(104_858)).toBe('0.1 MB');
  });

  it('never claims a file is 0 KB', () => {
    expect(fileSize(0)).toBe('1 KB');
  });
});

describe('duration', () => {
  it('hedges rather than inventing precision', () => {
    expect(duration(0)).toBe('a moment');
    expect(duration(-1)).toBe('a moment');
    expect(duration(Infinity)).toBe('a moment');
    expect(duration(59)).toBe('under a minute');
  });

  it('rounds to minutes, singular at one', () => {
    expect(duration(60)).toBe('about 1 minute');
    expect(duration(61)).toBe('about 1 minute');
    expect(duration(120)).toBe('about 2 minutes');
  });

  it('switches to hours at sixty minutes, singular at one', () => {
    expect(duration(3600)).toBe('about 1 hour');
    expect(duration(5400)).toBe('about 1h 30m');
    expect(duration(7200)).toBe('about 2 hours');
  });
});

describe('estimateSeconds', () => {
  it('is zero when there is nothing left', () => {
    expect(estimateSeconds(0, 4, [])).toBe(0);
  });

  it('falls back to the rate limit until two requests have been observed', () => {
    expect(estimateSeconds(4, 4, [])).toBe(60);
    expect(estimateSeconds(4, 4, [600])).toBe(60);
  });

  it('takes whichever of the two constraints binds harder', () => {
    // Fast requests: the per-minute limit is what will actually be felt.
    expect(estimateSeconds(4, 4, [30, 30])).toBe(60);
    // Slow requests: their own pace is.
    expect(estimateSeconds(4, 4, [120, 120])).toBe(120);
  });

  it('treats a pace of zero as one, not as a division by zero', () => {
    expect(estimateSeconds(4, 0, [])).toBe(240);
  });
});

describe('languageLabel', () => {
  it('names the undetected case rather than showing the wire value', () => {
    expect(languageLabel('auto')).toBe('Not detected');
    expect(languageLabel('')).toBe('Not detected');
  });

  it('title-cases what the server sends lowercase', () => {
    expect(languageLabel('chinese')).toBe('Chinese');
  });
});
