"""HTML table: the same books priced in the home market and a second market."""

from __future__ import annotations

import html


def _cell(m: dict | None) -> str:
    if not m or m.get("amount") is None:
        return "<span style='color:#999'>—</span>"
    conv = " (converted)" if m.get("converted") else ""
    return (f"<b>{m['amount']:.2f} {html.escape(m.get('currency', ''))}</b>{conv}<br><small><a href='{html.escape(m.get('url', ''))}'>"
            f"{html.escape(m.get('source', '')[:40])}</a> {html.escape(m.get('retrieved_at', '')[:10])}</small>")


def render(d: dict) -> str:
    h, s = d["home"], d["second"]
    rows = "".join(
        f"<tr><td>{html.escape(r['title'])}</td><td>{_cell(r['home_replacement'])}</td><td>{_cell(r['home_used'])}</td>"
        f"<td>{_cell(r['replacement'])}</td><td>{_cell(r['used'])}</td><td><small>{html.escape(r['appraisal'] or '; '.join(r['review'])[:120])}</small></td></tr>"
        for r in d["books"])
    return (f"<!doctype html><meta charset=utf-8><title>Locale comparison</title><style>body{{font:14px system-ui;margin:20px}}"
            f"td,th{{border-bottom:1px solid #ddd;padding:6px;text-align:left;vertical-align:top}}</style>"
            f"<h1>Same books, two markets</h1><p>Home: {h['country']} / {h['currency']} · Second: {s['country']} / {s['currency']}. "
            f"Locale is a runtime setting (<code>--country</code>); no prices or catalogue are hard-coded.</p>"
            f"<table><tr><th>Book</th><th>{h['country']} new</th><th>{h['country']} used</th><th>{s['country']} new</th>"
            f"<th>{s['country']} used</th><th>Notes</th></tr>{rows}</table>")
