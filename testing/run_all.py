"""
Run the suites.

    python testing/run_all.py               # everything, ~8 API requests
    python testing/run_all.py --only api    # only the free half
    python testing/run_all.py --only translation

Run it inside the app container, where Python and the dependencies are:

    docker compose exec -T epub-translate python /app/testing/run_all.py

Both suites retune `settings.env` as they go and put it back afterwards, so the
whole run happens under one override of its own: every upload leaves a job that
counts as active until it expires, and the capacity gate would otherwise start
refusing uploads part-way through a suite that makes a dozen of them.
"""

import argparse
import sys
import time

import fixtures
import test_api
import test_translation
import test_writer
from harness import Api, Report, run_tests, settings

SUITES = {
    'api': (test_api, "Free: uploading, planning, previewing, refusing"),
    'writer': (test_writer, "Free: what a book loses on its way back out"),
    'translation': (test_translation, "Paid: four real translation runs"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(SUITES), help="run one suite instead of both")
    parser.add_argument("--ip", default="198.51.100.7",
                        help="the address to arrive from; the cooldown is per address")
    arguments = parser.parse_args()

    api = Api(ip=arguments.ip)

    status = api.status()
    if status.status_code != 200:
        print(f"The server at {api.base_url} answered {status.status_code}. Is it running?")
        return 1
    if not status.json()['configured']:
        print("The server has no API key configured; there is nothing to test against.")
        return 1

    print(f"Server: {api.base_url}  model {status.json()['model']}  "
          f"{status.json()['remaining_requests']} of {status.json()['daily_budget']} "
          f"requests left today")

    books = fixtures.build_all()
    print(f"Fixtures: {', '.join(sorted(books))}")

    chosen = [arguments.only] if arguments.only else list(SUITES)
    failures = 0
    stamp = time.strftime("%Y%m%d-%H%M%S")

    # Raised for the length of the run: uploads accumulate as jobs that are
    # ready but not started, and every one of them counts against the capacity
    # gate. Lowering it applies at once, which is what `test_capacity_gate`
    # relies on; raising it only widens the admission gate, since the thread
    # pool was sized when the server started.
    with settings(MAX_TRANSLATIONS_AT_ONCE=64):
        for name in chosen:
            module, description = SUITES[name]
            report = Report(f"{name} — {description}")
            run_tests(report, module.TESTS, api, report, books)
            failures += report.summary()
            print(f"written to {report.save(f'{stamp}-{name}.json')}")

    spent = status.json()['remaining_requests'] - api.status().json()['remaining_requests']
    print(f"\nAPI requests spent by this run: {spent}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
