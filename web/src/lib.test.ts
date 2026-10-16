import { describe, expect, it } from "vitest";
import { decidedAt, tokenText, trustBand } from "./lib";

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
