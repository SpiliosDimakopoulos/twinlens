import type { ReactNode } from "react";

/** A numbered section of the study: title and lede on the left, figures on the right. */
export function Section({ n, title, children, lede }: { n: string; title: string; lede: ReactNode; children: ReactNode }) {
  return (
    <section className="grid gap-x-12 gap-y-4 border-t pb-10 pt-6 rule lg:grid-cols-[minmax(0,330px)_1fr]">
      <div>
        <div className="kicker">§ {n}</div>
        <h2 className="display mt-2 text-[28px] text-ink">{title}</h2>
        <div className="mt-3 space-y-3 text-[14.5px] leading-relaxed text-ink2">{lede}</div>
      </div>
      <div className="min-w-0 space-y-5">{children}</div>
    </section>
  );
}
