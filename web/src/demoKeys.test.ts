import { describe, expect, it } from "vitest";
import { nearest, presetKey } from "./demoKeys";

describe("nearest", () => {
  it("picks the closest value, the first one on a tie", () => {
    expect(nearest([0, 0.5, 2, 3], 1.1)).toBe(0.5);
    expect(nearest([0, 0.5, 2, 3], 2.6)).toBe(3);
    expect(nearest([0, 2], 1)).toBe(0);
    expect(nearest([0, 0.5, 2, 3], -4)).toBe(0);
  });
  it("returns undefined for an empty list", () => {
    expect(nearest([], 1)).toBeUndefined();
  });
});

describe("presetKey", () => {
  it("matches Python str(float)", () => {
    expect(presetKey(0)).toBe("0.0");
    expect(presetKey(2)).toBe("2.0");
    expect(presetKey(0.5)).toBe("0.5");
    expect(presetKey(0.25)).toBe("0.25");
  });
});
