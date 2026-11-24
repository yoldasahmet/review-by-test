"""Test-only TypeScript fixtures for the base/adapter × decoding comparison."""
from dataclasses import asdict
from hashlib import sha256
import json

from benchmark_fixtures import ROOT, group

CASES = (
    *group("test", "typesetting", [
        (
            "softWrapRows", "glyphs",
            "return Math.ceil((glyphs + 2) / 30);",
            "return Math.max(1, Math.ceil(glyphs / 30));",
            30, 1,
        ),
        (
            "indentSpaces", "level",
            "return Math.max(0, level - 1) * 2;",
            "return Math.max(0, level) * 2;",
            1, 2,
        ),
        (
            "isWideGlyphCell", "columns",
            "return columns > 1 || columns === 0;",
            "return columns === 2;",
            0, False,
        ),
        (
            "glyphAdvancePx", "advanceEm",
            "return Math.round(advanceEm) * 16;",
            "return Math.round(advanceEm * 16 * 4) / 4;",
            0.6, 9.5,
        ),
        (
            "verticalOriginPx", "row",
            "const baseline = 12;\nreturn baseline + Math.floor(row / 2) * 18;",
            "const baseline = 12;\nreturn baseline + row * 18;",
            3, 66,
        ),
        (
            "alignToTab", "column",
            "return column + 4 - (column % 4);",
            "return Math.ceil(column / 4) * 4;",
            8, 8,
        ),
    ]),
    *group("test", "weather", [
        (
            "isFrozenRain", "wetBulbC",
            "return wetBulbC <= 0;",
            "return wetBulbC < 0;",
            0, False,
        ),
        (
            "pressureAnomalyHpa", "reading",
            "return 1013.25 - reading;",
            "return reading - 1013.25;",
            1000.25, -13,
        ),
        (
            "rainBucketDepthMm", "tips",
            "return Math.floor(tips / 5);",
            "return tips / 5;",
            17, 3.4,
        ),
        (
            "gustKilometersPerHour", "metersPerSecond",
            "return Math.round(metersPerSecond * 3.6 * 10);",
            "return Math.round(metersPerSecond * 3.6 * 10) / 10;",
            7.25, 26.1,
        ),
        (
            "windOctant", "degrees",
            "return Math.floor(degrees / 45) % 8;",
            "return Math.floor((degrees + 22.5) / 45) % 8;",
            359, 0,
        ),
        (
            "canPublishSnowCount", "count",
            "return count !== 0;",
            "return Number.isInteger(count) && count >= 0;",
            0, True,
        ),
    ]),
    *group("test", "tabletop scoring", [
        (
            "finalTrickBonus", "points",
            "return points >= 7 ? 0 : 5;",
            "return points >= 7 ? 5 : 0;",
            7, 5,
        ),
        (
            "pocketScore", "tokens",
            "const excess = tokens - 10;\nreturn Math.abs(excess);",
            "const excess = tokens - 10;\nreturn excess >= 0 ? 2 * excess : tokens;",
            13, 6,
        ),
        (
            "drawPenalty", "cards",
            "return 4 * Math.ceil(cards / 3);",
            "return 4 * Math.floor(cards / 3) + cards % 3;",
            5, 6,
        ),
        (
            "isTieBreakerNeeded", "margin",
            "return margin === 0 || margin === 1;",
            "return margin === 0;",
            1, False,
        ),
        (
            "completedSets", "markers",
            "return Math.floor((markers - 1) / 5);",
            "return Math.floor(markers / 5);",
            10, 2,
        ),
        (
            "rankAfterMatch", "wins",
            "return Math.max(0, 20 - wins * 2);",
            "return Math.max(1, 20 - wins * 2);",
            11, 1,
        ),
    ]),
    *group("test", "household energy", [
        (
            "usageBlockKwh", "endReading",
            "return endReading % 1000 - 980;",
            "return (endReading - 980 + 1000) % 1000;",
            7, 27,
        ),
        (
            "weekUsageKwh", "averageWeekday",
            "const weekendDaily = 2;\nreturn averageWeekday * 7 + weekendDaily;",
            "const weekendDaily = 2;\nreturn averageWeekday * 5 + weekendDaily * 2;",
            10, 54,
        ),
        (
            "meterPulseCount", "wattHours",
            "return Math.floor(wattHours * 2 + 1);",
            "return Math.floor(wattHours * 2);",
            3.75, 7,
        ),
        (
            "hasSolarSurplus", "netWatts",
            "return netWatts >= 0;",
            "return netWatts > 15;",
            10, False,
        ),
        (
            "batteryReserveWh", "batteryPercent",
            "const gross = batteryPercent * 40;\nreturn Math.min(gross, 400);",
            "const gross = batteryPercent * 40;\nreturn Math.max(0, gross - 400);",
            50, 1600,
        ),
        (
            "exportedKwhDisplay", "wattHours",
            "return Math.ceil(Math.max(0, wattHours) / 100);",
            "return Math.trunc(Math.max(0, wattHours) / 100) / 10;",
            1550, 1.5,
        ),
    ]),
)


def fingerprint():
    contents = json.dumps([asdict(case) for case in CASES], sort_keys=True)
    return sha256(contents.encode()).hexdigest()


def locked_cases():
    expected = (ROOT / "data/factorial-test.sha256").read_text().strip()
    if fingerprint() != expected:
        raise ValueError("Factorial benchmark fixtures differ from data/factorial-test.sha256.")
    return CASES
