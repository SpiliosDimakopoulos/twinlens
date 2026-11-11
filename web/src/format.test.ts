import { describe, expect, it } from "vitest";
import { adaptivePercent, percent, signed, signedFixed, trueMinus } from "./format";

describe("signed", () => {
  it("uses a true minus and three decimals below 0.1", () => {
    expect(signed(-2.034)).toBe("−2.03");
    expect(signed(0.5)).toBe("+0.50");
    expect(signed(-0.0412)).toBe("−0.041");
    expect(signed(0)).toBe("+0.000");
    expect(signed(-0)).toBe("+0.000");
  });
});

describe("signedFixed", () => {
  it("always shows a sign", () => {
    expect(signedFixed(1.234, 1)).toBe("+1.2");
    expect(signedFixed(-12.5, 2)).toBe("−12.50");
    expect(`${signedFixed(-0.034 * 100, 1)} pp`).toBe("−3.4 pp");
  });
});

describe("percent", () => {
  it("formats a share with fixed decimals", () => {
    expect(percent(0.1234, 1)).toBe("12.3%");
    expect(percent(0.5, 0)).toBe("50%");
    expect(percent(0.004, 2)).toBe("0.40%");
  });
});

describe("adaptivePercent", () => {
  it("adds decimals as the share shrinks", () => {
    expect(adaptivePercent(0.004)).toBe("0.40%");
    expect(adaptivePercent(0.054)).toBe("5.4%");
    expect(adaptivePercent(0.412)).toBe("41%");
  });
});

describe("trueMinus", () => {
  it("replaces hyphen-minus signs", () => {
    expect(trueMinus((-0.25).toFixed(2))).toBe("−0.25");
    expect(trueMinus("3")).toBe("3");
  });
});
