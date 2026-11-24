"""Fresh, test-only fixtures for the constrained decoding comparison."""
from dataclasses import asdict
import hashlib
import json

from benchmark_fixtures import ROOT, group

CASES = (
    *group("test", "rail transit", [
        (
            "departureBoardMinute", "secondsAfterMidnight",
            "return Math.ceil(secondsAfterMidnight / 60) % 1440;",
            "return Math.floor(secondsAfterMidnight / 60) % 1440;",
            86399, 1439,
        ),
        (
            "isOffPeakMinute", "minute",
            "return minute >= 600 && minute <= 960;",
            "return minute >= 600 && minute < 960;",
            960, False,
        ),
        (
            "carriagesForRiders", "riders",
            "return Math.floor((riders + 79) / 80) + 1;",
            "return Math.max(1, Math.floor((riders + 79) / 80));",
            81, 2,
        ),
        (
            "transferCreditCents", "minutesSinceTap",
            "if (minutesSinceTap >= 30) return 0;\nreturn 250 - minutesSinceTap;",
            "if (minutesSinceTap >= 30) return 0;\nreturn 250;",
            15, 250,
        ),
        (
            "dispatchBufferSeconds", "dwellSeconds",
            "const buffer = dwellSeconds + 12;\nreturn Math.min(buffer, 20);",
            "const buffer = dwellSeconds + 12;\nreturn Math.max(buffer, 20);",
            3, 20,
        ),
        (
            "platformForTrain", "trainNumber",
            "return 1 + trainNumber % 3;",
            "return 1 + (trainNumber - 1) % 3;",
            6, 3,
        ),
    ]),
    *group("test", "greenhouse controls", [
        (
            "dualPumpOverlapSeconds", "phaseSeconds",
            "return 15 - Math.abs(phaseSeconds);",
            "return Math.max(0, 15 - Math.abs(phaseSeconds));",
            -18, 0,
        ),
        (
            "isVentOpen", "temperatureC",
            "return temperatureC < 18 || temperatureC > 27;",
            "return temperatureC > 27;",
            16, False,
        ),
        (
            "waterPerTrayMl", "cells",
            "const amount = cells * 7.5;\nreturn Math.round(amount / 5);",
            "const amount = cells * 7.5;\nreturn Math.round(amount / 5) * 5;",
            3, 25,
        ),
        (
            "filterServiceWeek", "daysSinceInstall",
            "return Math.floor(daysSinceInstall / 7);",
            "return 1 + Math.floor(daysSinceInstall / 7);",
            28, 5,
        ),
        (
            "lampDutyPercent", "daylightHours",
            "return Math.max(0, (16 - daylightHours) / 16);",
            "return Math.max(0, (16 - daylightHours) / 16) * 100;",
            12, 25,
        ),
        (
            "shouldIrrigate", "moisturePercent",
            "return moisturePercent < 35;",
            "return moisturePercent >= 0 && moisturePercent < 35;",
            -1, False,
        ),
    ]),
    *group("test", "cartography", [
        (
            "longitude360", "longitude",
            "return longitude % 360;",
            "return ((longitude % 360) + 360) % 360;",
            -45, 315,
        ),
        (
            "metersPerPixel", "zoom",
            "const worldMeters = 40000000;\nreturn worldMeters / (256 * (zoom + 1));",
            "const worldMeters = 40000000;\nreturn worldMeters / (256 * 2 ** zoom);",
            3, 19531.25,
        ),
        (
            "hemisphereIndex", "latitude",
            "return latitude < 0 ? 0 : 1;",
            "return latitude < 0 ? 1 : 0;",
            -20, 1,
        ),
        (
            "tileSpan", "zoom",
            "return 2 ** (zoom + 1);",
            "return 2 ** zoom;",
            5, 32,
        ),
        (
            "inPolarCap", "latitude",
            "return latitude >= 66.5;",
            "return Math.abs(latitude) >= 66.5;",
            -67, True,
        ),
        (
            "terrainColorBand", "elevationMeters",
            "const band = Math.floor(elevationMeters / 500);\nreturn Math.min(4, band);",
            "const band = Math.floor(elevationMeters / 500);\nreturn Math.max(0, Math.min(4, band));",
            -100, 0,
        ),
    ]),
    *group("test", "materials lab", [
        (
            "passesGauge", "thicknessMm",
            "return thicknessMm >= 1.9 || thicknessMm <= 2.1;",
            "return thicknessMm >= 1.9 && thicknessMm <= 2.1;",
            2.2, False,
        ),
        (
            "calibratedPressure", "readingKpa",
            "return (readingKpa + 10) * 1.25;",
            "return (readingKpa - 10) * 1.25;",
            30, 25,
        ),
        (
            "curingMinutes", "coats",
            "return 45 + 15 * coats;",
            "if (coats === 0) return 0;\nreturn 45 + 15 * coats;",
            0, 0,
        ),
        (
            "specimenMassGrams", "massMg",
            "return Math.round(massMg / 1000);",
            "return Math.round(massMg / 10) / 100;",
            3275, 3.28,
        ),
        (
            "dilutionMilliliters", "concentration",
            "const ratio = concentration / 20;\nreturn 50 * ratio;",
            "const ratio = concentration / 20;\nreturn 50 * (ratio - 1);",
            60, 100,
        ),
        (
            "compressionDeltaMm", "finalHeightMm",
            "return finalHeightMm - 25;",
            "return 25 - finalHeightMm;",
            20, 5,
        ),
    ]),
)


def fingerprint():
    contents = json.dumps([asdict(case) for case in CASES], sort_keys=True)
    return hashlib.sha256(contents.encode()).hexdigest()


def locked_cases():
    expected = (ROOT / "data/grammar-test.sha256").read_text().strip()
    if fingerprint() != expected:
        raise ValueError("Grammar benchmark fixtures differ from data/grammar-test.sha256.")
    return CASES
