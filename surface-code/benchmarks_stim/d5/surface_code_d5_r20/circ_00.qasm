OPENQASM 3;
include "stdgates.inc";
qubit[49] q;
bit[24] c;

// Rotated Surface Code d=5, 20 syndrome-extraction rounds
// Generated from Stim 1.15.0 (surface_code:rotated_memory_z)
// 25 data + 24 ancilla = 49 qubits
// X-stabilizers: 12, Z-stabilizers: 12
// CX schedule: 4 steps, 80 total CX per round

h q[0];
h q[1];
h q[7];
h q[8];
h q[10];
h q[14];
h q[22];
h q[24];

for int round in [0:20] {
  // Reset ancilla qubits
  reset q[25];
  reset q[26];
  reset q[27];
  reset q[28];
  reset q[29];
  reset q[30];
  reset q[31];
  reset q[32];
  reset q[33];
  reset q[34];
  reset q[35];
  reset q[36];
  reset q[37];
  reset q[38];
  reset q[39];
  reset q[40];
  reset q[41];
  reset q[42];
  reset q[43];
  reset q[44];
  reset q[45];
  reset q[46];
  reset q[47];
  reset q[48];

  // Hadamard on X-stabilizer ancillas
  h q[25];
  h q[26];
  h q[28];
  h q[30];
  h q[33];
  h q[35];
  h q[38];
  h q[40];
  h q[43];
  h q[45];
  h q[47];
  h q[48];

  // CX step 1
  cx q[25], q[1];
  cx q[33], q[11];
  cx q[43], q[21];
  cx q[28], q[7];
  cx q[38], q[17];
  cx q[26], q[3];
  cx q[35], q[13];
  cx q[45], q[23];
  cx q[30], q[9];
  cx q[40], q[19];
  cx q[10], q[32];
  cx q[20], q[42];
  cx q[6], q[27];
  cx q[16], q[37];
  cx q[12], q[34];
  cx q[22], q[44];
  cx q[8], q[29];
  cx q[18], q[39];
  cx q[14], q[36];
  cx q[24], q[46];

  // CX step 2
  cx q[25], q[0];
  cx q[33], q[10];
  cx q[43], q[20];
  cx q[28], q[6];
  cx q[38], q[16];
  cx q[26], q[2];
  cx q[35], q[12];
  cx q[45], q[22];
  cx q[30], q[8];
  cx q[40], q[18];
  cx q[5], q[32];
  cx q[15], q[42];
  cx q[1], q[27];
  cx q[11], q[37];
  cx q[7], q[34];
  cx q[17], q[44];
  cx q[3], q[29];
  cx q[13], q[39];
  cx q[9], q[36];
  cx q[19], q[46];

  // CX step 3
  cx q[33], q[6];
  cx q[43], q[16];
  cx q[28], q[2];
  cx q[38], q[12];
  cx q[47], q[22];
  cx q[35], q[8];
  cx q[45], q[18];
  cx q[30], q[4];
  cx q[40], q[14];
  cx q[48], q[24];
  cx q[5], q[27];
  cx q[15], q[37];
  cx q[11], q[34];
  cx q[21], q[44];
  cx q[7], q[29];
  cx q[17], q[39];
  cx q[13], q[36];
  cx q[23], q[46];
  cx q[9], q[31];
  cx q[19], q[41];

  // CX step 4
  cx q[33], q[5];
  cx q[43], q[15];
  cx q[28], q[1];
  cx q[38], q[11];
  cx q[47], q[21];
  cx q[35], q[7];
  cx q[45], q[17];
  cx q[30], q[3];
  cx q[40], q[13];
  cx q[48], q[23];
  cx q[0], q[27];
  cx q[10], q[37];
  cx q[6], q[34];
  cx q[16], q[44];
  cx q[2], q[29];
  cx q[12], q[39];
  cx q[8], q[36];
  cx q[18], q[46];
  cx q[4], q[31];
  cx q[14], q[41];

  // Undo Hadamard on X-stabilizer ancillas
  h q[25];
  h q[26];
  h q[28];
  h q[30];
  h q[33];
  h q[35];
  h q[38];
  h q[40];
  h q[43];
  h q[45];
  h q[47];
  h q[48];

  // Measure syndrome ancillas
  c[0] = measure q[27];
  c[1] = measure q[29];
  c[2] = measure q[31];
  c[3] = measure q[32];
  c[4] = measure q[34];
  c[5] = measure q[36];
  c[6] = measure q[37];
  c[7] = measure q[39];
  c[8] = measure q[41];
  c[9] = measure q[42];
  c[10] = measure q[44];
  c[11] = measure q[46];
  c[12] = measure q[25];
  c[13] = measure q[26];
  c[14] = measure q[28];
  c[15] = measure q[30];
  c[16] = measure q[33];
  c[17] = measure q[35];
  c[18] = measure q[38];
  c[19] = measure q[40];
  c[20] = measure q[43];
  c[21] = measure q[45];
  c[22] = measure q[47];
  c[23] = measure q[48];

  // Conditional corrections
  if (c[0]) {
    x q[6];
  }
  if (c[1]) {
    x q[8];
  }
  if (c[2]) {
    x q[9];
  }
  if (c[3]) {
    x q[10];
  }
  if (c[4]) {
    x q[12];
  }
  if (c[5]) {
    x q[14];
  }
  if (c[6]) {
    x q[16];
  }
  if (c[7]) {
    x q[18];
  }
  if (c[8]) {
    x q[19];
  }
  if (c[9]) {
    x q[20];
  }
  if (c[10]) {
    x q[22];
  }
  if (c[11]) {
    x q[24];
  }
  if (c[12]) {
    z q[1];
  }
  if (c[13]) {
    z q[3];
  }
  if (c[14]) {
    z q[7];
  }
  if (c[15]) {
    z q[9];
  }
  if (c[16]) {
    z q[11];
  }
  if (c[17]) {
    z q[13];
  }
  if (c[18]) {
    z q[17];
  }
  if (c[19]) {
    z q[19];
  }
  if (c[20]) {
    z q[21];
  }
  if (c[21]) {
    z q[23];
  }
  if (c[22]) {
    z q[22];
  }
  if (c[23]) {
    z q[24];
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
c[8] = measure q[8];
c[9] = measure q[9];
c[10] = measure q[10];
c[11] = measure q[11];
c[12] = measure q[12];
c[13] = measure q[13];
c[14] = measure q[14];
c[15] = measure q[15];
c[16] = measure q[16];
c[17] = measure q[17];
c[18] = measure q[18];
c[19] = measure q[19];
c[20] = measure q[20];
c[21] = measure q[21];
c[22] = measure q[22];
c[23] = measure q[23];