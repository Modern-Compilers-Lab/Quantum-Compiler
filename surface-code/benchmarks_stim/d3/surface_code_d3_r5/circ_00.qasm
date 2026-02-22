OPENQASM 3;
include "stdgates.inc";
qubit[17] q;
bit[8] c;

// Rotated Surface Code d=3, 5 syndrome-extraction rounds
// Generated from Stim 1.15.0 (surface_code:rotated_memory_z)
// 9 data + 8 ancilla = 17 qubits
// X-stabilizers: 4, Z-stabilizers: 4
// CX schedule: 4 steps, 24 total CX per round

h q[0];
h q[5];
h q[8];

for int round in [0:5] {
  // Reset ancilla qubits
  reset q[9];
  reset q[10];
  reset q[11];
  reset q[12];
  reset q[13];
  reset q[14];
  reset q[15];
  reset q[16];

  // Hadamard on X-stabilizer ancillas
  h q[9];
  h q[11];
  h q[14];
  h q[16];

  // CX step 1
  cx q[9], q[1];
  cx q[14], q[7];
  cx q[11], q[5];
  cx q[6], q[13];
  cx q[4], q[10];
  cx q[8], q[15];

  // CX step 2
  cx q[9], q[0];
  cx q[14], q[6];
  cx q[11], q[4];
  cx q[3], q[13];
  cx q[1], q[10];
  cx q[5], q[15];

  // CX step 3
  cx q[14], q[4];
  cx q[11], q[2];
  cx q[16], q[8];
  cx q[3], q[10];
  cx q[7], q[15];
  cx q[5], q[12];

  // CX step 4
  cx q[14], q[3];
  cx q[11], q[1];
  cx q[16], q[7];
  cx q[0], q[10];
  cx q[4], q[15];
  cx q[2], q[12];

  // Undo Hadamard on X-stabilizer ancillas
  h q[9];
  h q[11];
  h q[14];
  h q[16];

  // Measure syndrome ancillas
  c[0] = measure q[10];
  c[1] = measure q[12];
  c[2] = measure q[13];
  c[3] = measure q[15];
  c[4] = measure q[9];
  c[5] = measure q[11];
  c[6] = measure q[14];
  c[7] = measure q[16];

  // Conditional corrections
  if (c[0]) {
    x q[4];
  }
  if (c[1]) {
    x q[5];
  }
  if (c[2]) {
    x q[6];
  }
  if (c[3]) {
    x q[8];
  }
  if (c[4]) {
    z q[1];
  }
  if (c[5]) {
    z q[5];
  }
  if (c[6]) {
    z q[7];
  }
  if (c[7]) {
    z q[8];
  }

}

// Final data qubit readout
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
c[4] = measure q[4];
c[5] = measure q[5];
c[6] = measure q[6];
c[7] = measure q[7];