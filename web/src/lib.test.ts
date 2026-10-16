import { describe, expect, it } from "vitest";
import { decidedAt, tokenText, trustBand, verdict } from "./lib";

describe("tokenText", () => {
  it("keeps whitespace visible", () => {
    expect(tokenText("\n    return")).toBe(String.raw`\n    return`);
    expect(tokenText(" Paris")).toBe(" Paris");
    expect(tokenText("<|endoftext|>")).toBe("BOS");
  });
});

describe("decidedAt", () => {
  it("finds the first layer after which the lens never changes its mind", () => {
    expect(decidedAt({ token: " Rome", lens: [" the", " Rome", " the", " Rome", " Rome"] })).toBe(3);
    expect(decidedAt({ token: "x", lens: ["x", "x"] })).toBe(0);
  });
});

describe("trustBand", () => {
  it("maps scores to labelled bands", () => {
    expect(trustBand(0.9).status).toBe("good");
    expect(trustBand(0.5).status).toBe("warning");
    expect(trustBand(0.1).label).toMatch(/Don't trust/);
  });
});

describe("verdict", () => {
  it("calls out overstatement, sign flips and agreement", () => {
    expect(verdict(-2.03, -0.1, " basketball")).toMatch(/overstates the effect 20×/);
    expect(verdict(0.5, -0.4, " x")).toMatch(/other way/);
    expect(verdict(-0.5, -0.45, " x")).toMatch(/about right/);
    expect(verdict(0.001, 0.002, " x")).toMatch(/Neither/);
  });
});
