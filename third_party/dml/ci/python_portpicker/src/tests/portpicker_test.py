#!/usr/bin/python
#
# Copyright 2007 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Unittests for the portpicker module."""

from __future__ import print_function
import errno
import os
import random
import socket
import sys
import unittest

try:
    # pylint: disable=no-name-in-module
    from unittest import mock  # Python >= 3.3.
except ImportError:
    import mock  # https://pypi.python.org/pypi/mock

import portpicker


class PickUnusedPortTest(unittest.TestCase):
    def IsUnusedTCPPort(self, port):
        return self._bind(port, socket.SOCK_STREAM, socket.IPPROTO_TCP)

    def IsUnusedUDPPort(self, port):
        return self._bind(port, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    def setUp(self):
        # So we can Bind even if portpicker.bind is stubbed out.
        self._bind = portpicker.bind
        portpicker._owned_ports.clear()
        portpicker._free_ports.clear()
        portpicker._random_ports.clear()

    def testPickUnusedPortActuallyWorks(self):
        """This test can be flaky."""
        for _ in range(10):
            port = portpicker.pick_unused_port()
            self.assertTrue(self.IsUnusedTCPPort(port))
            self.assertTrue(self.IsUnusedUDPPort(port))

    @unittest.skipIf('PORTSERVER_ADDRESS' not in os.environ,
                     'no port server to test against')
    def testPickUnusedCanSuccessfullyUsePortServer(self):

        with mock.patch.object(portpicker, '_pick_unused_port_without_server'):
            portpicker._pick_unused_port_without_server.side_effect = (
                Exception('eek!')
            )

            # Since _PickUnusedPortWithoutServer() raises an exception, if we
            # can successfully obtain a port, the portserver must be working.
            port = portpicker.pick_unused_port()
            self.assertTrue(self.IsUnusedTCPPort(port))
            self.assertTrue(self.IsUnusedUDPPort(port))

    @unittest.skipIf('PORTSERVER_ADDRESS' not in os.environ,
                     'no port server to test against')
    def testPickUnusedCanSuccessfullyUsePortServerAddressKwarg(self):

        with mock.patch.object(portpicker, '_pick_unused_port_without_server'):
            portpicker._pick_unused_port_without_server.side_effect = (
                Exception('eek!')
            )

            # Since _PickUnusedPortWithoutServer() raises an exception, and
            # we've temporarily removed PORTSERVER_ADDRESS from os.environ, if
            # we can successfully obtain a port, the portserver must be working.
            addr = os.environ.pop('PORTSERVER_ADDRESS')
            try:
                port = portpicker.pick_unused_port(portserver_address=addr)
                self.assertTrue(self.IsUnusedTCPPort(port))
                self.assertTrue(self.IsUnusedUDPPort(port))
            finally:
              os.environ['PORTSERVER_ADDRESS'] = addr

    @unittest.skipIf('PORTSERVER_ADDRESS' not in os.environ,
                     'no port server to test against')
    def testGetPortFromPortServer(self):
        """Exercise the get_port_from_port_server() helper function."""
        for _ in range(10):
            port = portpicker.get_port_from_port_server(
                os.environ['PORTSERVER_ADDRESS'])
            self.assertTrue(self.IsUnusedTCPPort(port))
            self.assertTrue(self.IsUnusedUDPPort(port))

    def testSendsPidToPortServer(self):
        server = mock.Mock()
        server.recv.return_value = b'42768\n'
        with mock.patch.object(socket, 'socket', return_value=server):
            port = portpicker.get_port_from_port_server('portserver', pid=1234)
            server.sendall.assert_called_once_with(b'1234\n')
        self.assertEqual(port, 42768)

    def testPidDefaultsToOwnPid(self):
        server = mock.Mock()
        server.recv.return_value = b'52768\n'
        with mock.patch.object(socket, 'socket', return_value=server):
            with mock.patch.object(os, 'getpid', return_value=9876):
                port = portpicker.get_port_from_port_server('portserver')
                server.sendall.assert_called_once_with(b'9876\n')
        self.assertEqual(port, 52768)

    @mock.patch.dict(os.environ,{'PORTSERVER_ADDRESS': 'portserver'})
    def testReusesPortServerPorts(self):
        server = mock.Mock()
        server.recv.side_effect = [b'12345\n', b'23456\n', b'34567\n']
        with mock.patch.object(socket, 'socket', return_value=server):
            self.assertEqual(portpicker.pick_unused_port(), 12345)
            self.assertEqual(portpicker.pick_unused_port(), 23456)
            portpicker.return_port(12345)
            self.assertEqual(portpicker.pick_unused_port(), 12345)

    @mock.patch.dict(os.environ,{'PORTSERVER_ADDRESS': ''})
    def testDoesntReuseRandomPorts(self):
        ports = set()
        for _ in range(10):
            try:
                port = portpicker.pick_unused_port()
            except portpicker.NoFreePortFoundError:
                # This sometimes happens when not using portserver. Just
                # skip to the next attempt.
                continue
            ports.add(port)
            portpicker.return_port(port)
        self.assertGreater(len(ports), 5)  # Allow some random reuse.

    def testReturnsReservedPorts(self):
        with mock.patch.object(portpicker, '_pick_unused_port_without_server'):
            portpicker._pick_unused_port_without_server.side_effect = (
                Exception('eek!'))
            # Arbitrary port. In practice you should get this from somewhere
            # that assigns ports.
            reserved_port = 28465
            portpicker.add_reserved_port(reserved_port)
            ports = set()
            for _ in range(10):
                port = portpicker.pick_unused_port()
                ports.add(port)
                portpicker.return_port(port)
            self.assertEqual(len(ports), 1)
            self.assertEqual(ports.pop(), reserved_port)

    @mock.patch.dict(os.environ,{'PORTSERVER_ADDRESS': ''})
    def testFallsBackToRandomAfterRunningOutOfReservedPorts(self):
        # Arbitrary port. In practice you should get this from somewhere
        # that assigns ports.
        reserved_port = 23456
        portpicker.add_reserved_port(reserved_port)
        self.assertEqual(portpicker.pick_unused_port(), reserved_port)
        self.assertNotEqual(portpicker.pick_unused_port(), reserved_port)

    def testRandomlyChosenPorts(self):
        # Unless this box is under an overwhelming socket load, this test
        # will heavily exercise the "pick a port randomly" part of the
        # port picking code, but may never hit the "OS assigns a port"
        # code.
        ports = 0
        for _ in range(100):
            try:
                port = portpicker._pick_unused_port_without_server()
            except portpicker.NoFreePortFoundError:
                # Without the portserver, pick_unused_port can sometimes fail
                # to find a free port. Check that it passes most of the time.
                continue
            self.assertTrue(self.IsUnusedTCPPort(port))
            self.assertTrue(self.IsUnusedUDPPort(port))
            ports += 1
        # Getting a port shouldn't have failed very often, even on machines
        # with a heavy socket load.
        self.assertGreater(ports, 95)

    def testOSAssignedPorts(self):
        self.last_assigned_port = None

        def error_for_explicit_ports(port, socket_type, socket_proto):
            # Only successfully return a port if an OS-assigned port is
            # requested, or if we're checking that the last OS-assigned port
            # is unused on the other protocol.
            if port == 0 or port == self.last_assigned_port:
                self.last_assigned_port = self._bind(port, socket_type,
                                                     socket_proto)
                return self.last_assigned_port
            else:
                return None

        with mock.patch.object(portpicker, 'bind', error_for_explicit_ports):
            # Without server, this can be little flaky, so check that it
            # passes most of the time.
            ports = 0
            for _ in range(100):
                try:
                    port = portpicker._pick_unused_port_without_server()
                except portpicker.NoFreePortFoundError:
                    continue
                self.assertTrue(self.IsUnusedTCPPort(port))
                self.assertTrue(self.IsUnusedUDPPort(port))
                ports += 1
            self.assertGreater(ports, 95)

    def pickUnusedPortWithoutServer(self):
        # Try a few times to pick a port, to avoid flakiness and to make sure
        # the code path we want was exercised.
        for _ in range(5):
            try:
                port = portpicker._pick_unused_port_without_server()
            except portpicker.NoFreePortFoundError:
                continue
            else:
                self.assertTrue(self.IsUnusedTCPPort(port))
                self.assertTrue(self.IsUnusedUDPPort(port))
                return
        self.fail("Failed to find a free port")

    def testPickPortsWithoutServer(self):
        # Test the first part of _pick_unused_port_without_server, which
        # tries a few random ports and checks is_port_free.
        self.pickUnusedPortWithoutServer()

        # Now test the second part, the fallback from above, which asks the
        # OS for a port.
        def mock_port_free(port):
            return False

        with mock.patch.object(portpicker, 'is_port_free', mock_port_free):
            self.pickUnusedPortWithoutServer()

    def checkIsPortFree(self):
        """This might be flaky unless this test is run with a portserver."""
        # The port should be free initially.
        port = portpicker.pick_unused_port()
        self.assertTrue(portpicker.is_port_free(port))

        cases = [
            (socket.AF_INET,  socket.SOCK_STREAM, None),
            (socket.AF_INET6, socket.SOCK_STREAM, 0),
            (socket.AF_INET6, socket.SOCK_STREAM, 1),
            (socket.AF_INET,  socket.SOCK_DGRAM,  None),
            (socket.AF_INET6, socket.SOCK_DGRAM,  0),
            (socket.AF_INET6, socket.SOCK_DGRAM,  1),
        ]
        for (sock_family, sock_type, v6only) in cases:
            # Occupy the port on a subset of possible protocols.
            try:
                sock = socket.socket(sock_family, sock_type, 0)
            except socket.error:
                print('Kernel does not support sock_family=%d' % sock_family,
                      file=sys.stderr)
                # Skip this case, since we cannot occupy a port.
                continue

            if not hasattr(socket, 'IPPROTO_IPV6'):
                v6only = None

            if v6only is not None:
                try:
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY,
                                    v6only)
                except socket.error:
                    print('Kernel does not support IPV6_V6ONLY=%d' % v6only,
                          file=sys.stderr)
                    # Don't care; just proceed with the default.

            # Socket may have been taken in the mean time, so catch the
            # socket.error with errno set to EADDRINUSE and skip this
            # attempt.
            try:
                sock.bind(('', port))
            except socket.error as e:
                if e.errno == errno.EADDRINUSE:
                    raise portpicker.NoFreePortFoundError
                raise

            # The port should be busy.
            self.assertFalse(portpicker.is_port_free(port))
            sock.close()

            # Now it's free again.
            self.assertTrue(portpicker.is_port_free(port))

    def testIsPortFree(self):
        # This can be quite flaky on a busy host, so try a few times.
        for _ in range(10):
            try:
                self.checkIsPortFree()
            except portpicker.NoFreePortFoundError:
                pass
            else:
                return
        self.fail("checkPortIsFree failed every time.")

    def testIsPortFreeException(self):
        port = portpicker.pick_unused_port()
        with mock.patch.object(socket, 'socket') as mock_sock:
            mock_sock.side_effect = socket.error('fake socket error', 0)
            self.assertFalse(portpicker.is_port_free(port))

    def testThatLegacyCapWordsAPIsExist(self):
        """The original APIs were CapWords style, 1.1 added PEP8 names."""
        self.assertEqual(portpicker.bind, portpicker.Bind)
        self.assertEqual(portpicker.is_port_free, portpicker.IsPortFree)
        self.assertEqual(portpicker.pick_unused_port, portpicker.PickUnusedPort)
        self.assertEqual(portpicker.get_port_from_port_server,
                         portpicker.GetPortFromPortServer)


if __name__ == '__main__':
    unittest.main()
