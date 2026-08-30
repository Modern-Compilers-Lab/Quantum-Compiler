OPENQASM 3;
include "stdgates.inc";

qubit[16] q;
bit[6] c;

h q[6];

measure q[6] -> c[0];
while (c[0] == false) {

  measure q[4] -> c[1];
  if (c[1] == true) {
    cx q[12], q[7];
    x q[7];
    h q[12];
    rz(0.78539816) q[7];
  } else {
    cx q[13], q[0];
    h q[1];
    rz(2.74889357) q[1];
    x q[0];
  }

  measure q[2] -> c[2];
  if (c[2] == true) {
    cx q[9], q[3];
    h q[3];
    rz(1.57079633) q[9];
    cx q[12], q[7];
  } else {
    h q[4];
    cx q[13], q[0];
    x q[4];
    rz(0.78539816) q[0];
  }

  measure q[12] -> c[3];
  if (c[3] == true) {
    cx q[15], q[2];
    x q[14];
    cx q[0], q[14];
    h q[2];
  } else {
    cx q[12], q[10];
    x q[5];
    h q[10];
    rz(1.04719755) q[5];
  }

  measure q[8] -> c[4];
  if (c[4] == true) {
    cx q[15], q[12];
    h q[7];
    x q[12];
    rz(1.57079633) q[7];
  } else {
    cx q[14], q[3];
    h q[0];
    cx q[11], q[0];
    x q[3];
  }

  measure q[12] -> c[5];
  if (c[5] == true) {
    cx q[4], q[3];
    rz(2.74889357) q[6];
    x q[14];
    h q[6];
  } else {
    cx q[13], q[1];
    x q[11];
    rz(1.96349541) q[7];
    rx(1.57079633) q[6];
  }

  h q[6];
  measure q[6] -> c[0];
}
