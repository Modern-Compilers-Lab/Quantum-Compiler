OPENQASM 3;
include "stdgates.inc";

qubit[16] q;
bit[2] c;

h q[5];

measure q[5] -> c[0];
while (c[0] == false) {

  cx q[10], q[0];
  rz(1.96349541) q[4];
  x q[0];
  rz(2.74889357) q[14];

  h q[0];

  measure q[0] -> c[1];
  while (c[1] == false) {

    cx q[15], q[2];
    rz(1.96349541) q[12];
    h q[2];
    x q[12];

    cx q[13], q[0];
    rz(0.39269908) q[1];
    x q[0];
    rz(1.96349541) q[1];

    cx q[15], q[12];
    rz(2.74889357) q[7];
    h q[12];
    x q[7];

    h q[0];
    measure q[0] -> c[1];
  }

  cx q[15], q[11];
  cx q[0], q[8];
  h q[11];
  x q[8];

  h q[5];
  measure q[5] -> c[0];
}
