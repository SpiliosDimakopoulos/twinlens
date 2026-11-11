/** Edit-strength presets of the what-if panel: button label and activation multiplier. */
export const PRESETS: [string, number][] = [["Off", 0], ["½×", 0.5], ["2×", 2], ["3×", 3]];

/** Button tooltip per preset multiplier. */
export const PRESET_HINT: Record<number, string> = { 0: "Switch the feature off", 0.5: "Halve its strength", 2: "Double its strength", 3: "Triple its strength" };

/** Verb for the panel title ("What if we <verb> this feature?") per preset multiplier. */
export const PRESET_VERB: Record<number, string> = { 0: "remove", 0.5: "halve", 2: "double", 3: "triple" };
