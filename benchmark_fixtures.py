"""Authored TypeScript fixes and reference witnesses for the locked benchmark."""
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from fixtures import diff, reference

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Case:
    split: str
    domain: str
    name: str
    before: str
    after: str
    argument: int | float
    expected: int | float | bool

    @property
    def patch(self):
        return diff(self.before, self.after)

    @property
    def reference(self):
        return reference(self.name, self.argument, self.expected)


def group(split, domain, rows):
    for name, parameter, before, after, argument, expected in rows:
        def source(body):
            lines = "\n".join(f"  {line}" for line in body.splitlines())
            return f"export function {name}({parameter}: number) {{\n{lines}\n}}\n"

        yield Case(split, domain, name, source(before), source(after), argument, expected)


CASES = (
    *group("train", "fulfillment", [
        ("requiresFreight", "weightKg", "return weightKg > 70;", "return weightKg >= 70;", 70, True),
        ("isLetterSize", "thicknessMm", "return thicknessMm < 5;", "return thicknessMm <= 5;", 5, True),
        ("canUseLocker", "heightCm", "return heightCm <= 42;", "return heightCm < 42;", 42, False),
        ("palletsRequired", "cartons", "return Math.floor(cartons / 48);", "return Math.ceil(cartons / 48);", 49, 2),
        ("billableWeight", "weightKg", "return Math.min(1, weightKg);", "return Math.max(1, weightKg);", 0.25, 1),
        ("remainingPickupSlots", "booked", "return 12 - booked;", "return Math.max(0, 12 - booked);", 15, 0),
        ("crateCharge", "crates", "return crates * 6 + 15;", "return crates === 0 ? 0 : crates * 6 + 15;", 0, 0),
    ]),
    *group("train", "calendar", [
        ("canBookAhead", "days", "return days <= 90;", "return days >= 0 && days <= 90;", -1, False),
        ("weekdayIndex", "day", "return day % 7;", "return ((day % 7) + 7) % 7;", -2, 5),
        ("hoursToSeconds", "hours", "return hours * 60;", "return hours * 60 * 60;", 0.5, 1800),
        ("hourOfDay", "minutes", "return Math.floor(minutes / 60);", "return Math.floor(minutes / 60) % 24;", 1500, 1),
        ("isLeapYear", "year", "return year % 4 === 0;", "return year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);", 2100, False),
        ("signedMinuteOffset", "zonesEast", "return Math.abs(zonesEast) * 60;", "return zonesEast * 60;", -3, -180),
        ("isWeekend", "isoDay", "return isoDay === 6 && isoDay === 7;", "return isoDay === 6 || isoDay === 7;", 7, True),
    ]),
    *group("train", "telemetry", [
        ("stepsPerMinute", "steps", "return steps * 2;", "return steps / 2;", 140, 70),
        ("fahrenheitToCelsius", "fahrenheit", "return fahrenheit * 5 / 9;", "return (fahrenheit - 32) * 5 / 9;", 50, 10),
        ("normalizedBrightness", "level", "return level / 100;", "return Math.min(100, Math.max(0, level)) / 100;", 125, 1),
        ("isMissingReading", "reading", "return reading <= 0;", "return reading === -999;", 0, False),
        ("absoluteDrift", "reading", "return reading - 20;", "return Math.abs(reading - 20);", 16, 4),
        ("sampleBytes", "samples", "return samples + 4;", "return samples * 4;", 3, 12),
        ("nominalVoltage", "millivolts", "return millivolts / 100;", "return millivolts / 1000;", 3300, 3.3),
    ]),
    *group("train", "library", [
        ("renewalsRemaining", "used", "return Math.min(0, 2 - used);", "return Math.max(0, 2 - used);", 1, 1),
        ("standardLoanDays", "isStaffFlag", "return isStaffFlag ? 21 : 42;", "return isStaffFlag ? 42 : 21;", 0, 21),
        ("isMemberTier", "tier", "return tier >= 1 || tier <= 3;", "return tier >= 1 && tier <= 3;", 4, False),
        ("archiveBox", "catalogNumber", "return Math.floor(catalogNumber / 12);", "return Math.floor((catalogNumber - 1) / 12);", 24, 1),
        ("bundleWeightGrams", "volumes", "return (volumes + 1) * 300;", "return volumes * 300 + 80;", 2, 680),
        ("isLargePrintCode", "code", "return code % 2 === 0;", "return code % 2 !== 0;", 13, True),
        ("fineCapCents", "daysLate", "return Math.min(200, daysLate) * 25;", "return Math.min(200, daysLate * 25);", 10, 200),
    ]),
    *group("valid", "subscriptions", [
        ("prorateCents", "daysLeft", "const share = daysLeft / 30;\nreturn Math.floor(500 * share);", "const share = daysLeft / 30;\nreturn Math.round(500 * share);", 4, 67),
        ("isTrialActive", "elapsedDays", "return elapsedDays !== 14;", "return elapsedDays < 14;", 15, False),
        ("seatOverageCents", "seats", "const overage = seats - 3;\nreturn overage * 250;", "const overage = seats - 3;\nreturn overage > 0 ? overage * 250 : 0;", 2, 0),
        ("annualCyclesCovered", "monthsPaid", "return Math.min(1, Math.floor(monthsPaid / 12));", "return Math.floor(monthsPaid / 12);", 25, 2),
    ]),
    *group("valid", "analytics", [
        ("activeRatio", "activeUsers", "const cohortSize = 200;\nreturn (cohortSize - activeUsers) / cohortSize;", "const cohortSize = 200;\nreturn activeUsers / cohortSize;", 50, 0.25),
        ("evenHistogramBin", "count", "const bin = Math.floor(count / 5);\nreturn bin % 2 === 1;", "const bin = Math.floor(count / 5);\nreturn bin % 2 === 0;", 11, True),
        ("lastCompleteHour", "elapsedMinutes", "return Math.trunc(elapsedMinutes / 60);", "return Math.floor(elapsedMinutes / 60);", -1, -1),
        ("nextSamplePowerOfTwo", "samples", "const exponent = Math.floor(Math.log2(samples));\nreturn 2 ** exponent;", "const exponent = Math.ceil(Math.log2(samples));\nreturn 2 ** exponent;", 9, 16),
    ]),
    *group("test", "cursor pagination", [
        ("inclusiveRowsLeft", "lastIndex", "const lastAvailable = 99;\nreturn lastAvailable - lastIndex;", "const lastAvailable = 99;\nreturn lastAvailable - lastIndex + 1;", 99, 1),
        ("cursorExpiresAt", "createdAtSeconds", "const lifetimeMinutes = 15;\nreturn (createdAtSeconds + lifetimeMinutes) * 60;", "const lifetimeMinutes = 15;\nreturn createdAtSeconds + lifetimeMinutes * 60;", 10, 910),
        ("shouldFetchAnotherBatch", "batchSize", "const limit = 25;\nreturn batchSize > 0 && batchSize !== limit;", "const limit = 25;\nreturn batchSize === limit;", 25, True),
        ("clampEmptyCursor", "cursor", "return cursor || 1;", "return cursor < 0 ? 1 : cursor;", 0, 0),
        ("resumeIndex", "encodedCursor", "// The cursor stores the last row already seen.\nreturn encodedCursor;", "// The cursor stores the last row already seen.\nreturn encodedCursor + 1;", 19, 20),
    ]),
    *group("test", "media processing", [
        ("padToEvenWidth", "pixels", "return pixels + 2 - (pixels % 2);", "return pixels + (pixels % 2);", 640, 640),
        ("hasRenderableFrames", "frameCount", "return Number.isInteger(frameCount);", "return Number.isInteger(frameCount) && frameCount > 0;", 0, False),
        ("isSupportedAudioRate", "hertz", "return hertz >= 44100 && hertz <= 48000;", "return hertz === 44100 || hertz === 48000;", 45000, False),
        ("previewWidth", "sourceWidth", "const scale = 0.25;\nreturn Math.round(sourceWidth) * scale;", "const scale = 0.25;\nreturn Math.round(sourceWidth * scale);", 1278, 320),
        ("transparentPixelAlpha", "opacityPercent", "if (opacityPercent === 0) return 255;\nreturn Math.round(opacityPercent * 255 / 100);", "if (opacityPercent === 0) return 0;\nreturn Math.round(opacityPercent * 255 / 100);", 0, 0),
    ]),
)


def fingerprint():
    contents = json.dumps([asdict(case) for case in CASES], sort_keys=True)
    return hashlib.sha256(contents.encode()).hexdigest()


def locked_cases():
    expected = (ROOT / "data/benchmark.sha256").read_text().strip()
    if fingerprint() != expected:
        raise ValueError("Benchmark fixtures differ from data/benchmark.sha256.")
    return CASES
