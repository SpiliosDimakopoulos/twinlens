import { afterEach, describe, expect, it, vi } from "vitest";
import { readParam, writeParams } from "./url";

function stubLocation(search: string, pathname = "/twinlens/", hash = "") {
  const replaceState = vi.fn();
  vi.stubGlobal("location", { search, pathname, hash });
  vi.stubGlobal("history", { replaceState });
  return replaceState;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("readParam", () => {
  it("reads a query parameter", () => {
    stubLocation("?p=2&f=3-4-5");
    expect(readParam("p")).toBe("2");
    expect(readParam("f")).toBe("3-4-5");
    expect(readParam("missing")).toBeNull();
  });
  it("returns null when location is unavailable", () => {
    vi.stubGlobal("location", undefined);
    expect(readParam("p")).toBeNull();
  });
});

describe("writeParams", () => {
  it("sets, replaces and deletes parameters, keeping path and hash", () => {
    const replaceState = stubLocation("?p=1&q=old", "/twinlens/", "#main");
    writeParams({ p: 3, q: null, m: 0.5, view: "" });
    expect(replaceState).toHaveBeenCalledWith(null, "", "/twinlens/?p=3&m=0.5#main");
  });
  it("drops the question mark when no parameters are left", () => {
    const replaceState = stubLocation("?view=study", "/");
    writeParams({ view: undefined });
    expect(replaceState).toHaveBeenCalledWith(null, "", "/");
  });
  it("fails silently when history is unavailable", () => {
    vi.stubGlobal("location", { search: "", pathname: "/", hash: "" });
    vi.stubGlobal("history", undefined);
    expect(() => writeParams({ p: 1 })).not.toThrow();
  });
});
