#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Parses an AbslTest XML log to produce a flattened summary of test results with the following structure:
{
    "Counts":
    {
        "Passed": [int]
        "Failed": [int]
        "Skipped": [int]
        "Blocked": [int]
        "Total": [int]
    }

    "Tests":
    [
        {
            "Name": [string]
            "Result": [string] (one of "Passed", "Failed", "Skipped", "Blocked")
            "Time" : [double] (time to execute in seconds)
            "Errors": [string] (optional; only present if result is "fail")
        }
        ...
    ]

    "Time" : [double] (time for all modules to finish, in seconds; can be smaller than the sum of test times, because of parallel execution)
    "Errors": [string] (optional; only present if errors occur outside of test scopes)
}
"""

import collections
import re
import argparse
import xml.etree.ElementTree as ET
import json
import glob
import os

CrashedTestInfo = collections.namedtuple('CrashedTestInfo', 'name path error')


def _parse_args():
  """Parses the arguments given to this script."""

  parser = argparse.ArgumentParser()
  parser.add_argument("--log_path",
                      help="path of the test's text file log",
                      required=True)

  parser.add_argument("--out_summary_path",
                      help="path to the summary json file to output",
                      required=True)

  parser.add_argument("--out_error_log_path",
                      help="path to the error log to output",
                      required=True)

  parser.add_argument("--xml_files_dir",
                      help="paths of directory where the xml files are located",
                      required=True)

  return parser.parse_args()


def _parse_test_crashes(log_path):
  test_crashes = []

  path_pattern = re.compile(r"Running '(.+)' with shard \d+\.\.\.")

  with open(log_path, 'r', encoding='utf-16') as test_log:
    line = test_log.readline()

    current_test_title = None
    current_test_path = ''
    error_lines = []

    while line:
      if line.startswith('[ RUN      ]'):
        if current_test_title is not None:
          test_info = CrashedTestInfo(name=current_test_title,
                                      path=current_test_path,
                                      error=''.join(error_lines))
          test_crashes.append(test_info)
          current_test_title = None
          error_lines.clear()

        current_test_title = line[line.index(']') + 2:].rstrip('\n')
      else:
        if (line.startswith('[       OK ]') or
            line.startswith('[  FAILED  ]') or line.startswith('[  SKIPPED ]')):
          current_test_title = None
          error_lines.clear()
        else:
          path_matches = path_pattern.match(line)

          if path_matches is not None:
            if current_test_title is not None:
              test_info = CrashedTestInfo(name=current_test_title,
                                          path=current_test_path,
                                          error=''.join(error_lines))
              test_crashes.append(test_info)
              current_test_title = None
              error_lines.clear()

            current_test_path = path_matches.group(1)
          else:
            error_lines.append(line)

      line = test_log.readline()

    if current_test_title is not None:
      test_info = CrashedTestInfo(name=current_test_title,
                                  path=current_test_path,
                                  error=''.join(error_lines))
      test_crashes.append(test_info)

  return test_crashes


def _generate_test_summary(xml_files_dir, test_crashes):
  xml_paths = glob.glob(os.path.join(xml_files_dir, '**', '*_test_result.xml'),
                        recursive=True)

  test_summary = {}
  test_summary['Time'] = 0.0
  test_summary['Tests'] = []
  test_summary['Counts'] = {}
  test_summary['Counts']['Total'] = 0
  test_summary['Counts']['Passed'] = 0
  test_summary['Counts']['Failed'] = 0
  test_summary['Counts']['Skipped'] = 0
  test_summary['Counts']['Blocked'] = 0

  for xml_path in xml_paths:
    # The module name is encoded in the name of the file
    module = re.sub(r'(.+)_test_result\.xml', lambda match: match.group(1),
                    xml_path)

    try:
      root = ET.parse(xml_path).getroot()
    except ET.ParseError:
      print('Skipping empty XML')
      continue

    # Since all processes start at the same time, the total execution time is
    # the time it took for the slowest one to finish
    test_summary['Time'] = max(test_summary['Time'], float(root.attrib['time']))

    for test_suite in root.findall('testsuite'):
      test_suite_name = test_suite.attrib['name']

      for test_case in test_suite.findall('testcase'):
        test_case_name = test_case.attrib['name']
        json_test_case = {}
        json_test_case['Name'] = f'{test_suite_name}.{test_case_name}'
        json_test_case['Module'] = module
        json_test_case['Time'] = test_case.attrib['time']

        # Failures are saved as children nodes instead of attributes
        failures = test_case.findall('failure') + test_case.findall('error')

        if failures:
          json_test_case['Result'] = 'Fail'
          test_summary['Counts']['Failed'] += 1
          error_strings = []

          for failure in failures:
            failure_message = failure.attrib['message']

            if re.match(r'.+\.(cpp|h):\d+', failure_message) is None:
              error_strings.append(failure_message)
            else:
              file_path = re.sub(r'(.+):\d+', lambda match: match.group(1),
                                 failure_message)
              line_number = re.sub(r'.+:(\d+)', lambda match: match.group(1),
                                   failure_message)
              message = re.sub(r'&#xA(.+)', lambda match: match.group(1),
                               failure_message)
              error_strings.append(f'{message} [{file_path}:{line_number}]')

          json_test_case['Errors'] = ''.join(error_strings).replace(
              '&#xA', '     ')
        else:
          try:
            status = test_case.attrib['status']
            pass_string = 'run'
            skip_string = 'skipped'
          except KeyError:
            status = test_case.attrib['result']
            pass_string = 'completed'
            skip_string = 'suppressed'

          if status == pass_string:
            json_test_case['Result'] = 'Pass'
            test_summary['Counts']['Passed'] += 1
          elif status == skip_string:
            json_test_case['Result'] = 'Skipped'
            test_summary['Counts']['Skipped'] += 1
          else:
            json_test_case['Result'] = 'Blocked'
            test_summary['Counts']['Blocked'] += 1

        test_summary['Counts']['Total'] += 1
        test_summary['Tests'].append(json_test_case)

  return test_summary


def main():
  args = _parse_args()
  test_crashes = _parse_test_crashes(args.log_path)
  test_summary = _generate_test_summary(args.xml_files_dir, test_crashes)

  with open(args.out_summary_path, 'w') as outfile:
    json.dump(test_summary, outfile)

  if 'Errors' in test_summary:
    with open(args.out_error_log_path, 'w') as outfile:
      outfile.write(test_summary['Errors'])


if __name__ == "__main__":
  main()
