OPENQASM 3;
include "stdgates.inc";
qubit[97] q;
bit[48] c;

// Rotated Surface Code d=7, 3 syndrome-extraction rounds
// Generated from Stim 1.15.0 (surface_code:rotated_memory_z)
// 49 data + 48 ancilla = 97 qubits
// X-stabilizers: 24, Z-stabilizers: 24
// CX schedule: 4 steps, 168 total CX per round

h q[0];
h q[2];
h q[3];
h q[9];
h q[10];
h q[14];
h q[15];
h q[16];
h q[27];
h q[28];
h q[31];
h q[34];
h q[36];
h q[39];
h q[40];
h q[42];

for int round in [0:3] {
  // Reset ancilla qubits
  reset q[49];
  reset q[50];
  reset q[51];
  reset q[52];
  reset q[53];
  reset q[54];
  reset q[55];
  reset q[56];
  reset q[57];
  reset q[58];
  reset q[59];
  reset q[60];
  reset q[61];
  reset q[62];
  reset q[63];
  reset q[64];
  reset q[65];
  reset q[66];
  reset q[67];
  reset q[68];
  reset q[69];
  reset q[70];
  reset q[71];
  reset q[72];
  reset q[73];
  reset q[74];
  reset q[75];
  reset q[76];
  reset q[77];
  reset q[78];
  reset q[79];
  reset q[80];
  reset q[81];
  reset q[82];
  reset q[83];
  reset q[84];
  reset q[85];
  reset q[86];
  reset q[87];
  reset q[88];
  reset q[89];
  reset q[90];
  reset q[91];
  reset q[92];
  reset q[93];
  reset q[94];
  reset q[95];
  reset q[96];

  // Hadamard on X-stabilizer ancillas
  h q[49];
  h q[50];
  h q[51];
  h q[53];
  h q[55];
  h q[57];
  h q[60];
  h q[62];
  h q[64];
  h q[67];
  h q[69];
  h q[71];
  h q[74];
  h q[76];
  h q[78];
  h q[81];
  h q[83];
  h q[85];
  h q[88];
  h q[90];
  h q[92];
  h q[94];
  h q[95];
  h q[96];

  // CX step 1
  cx q[49], q[1];
  cx q[60], q[15];
  cx q[74], q[29];
  cx q[88], q[43];
  cx q[53], q[9];
  cx q[67], q[23];
  cx q[81], q[37];
  cx q[50], q[3];
  cx q[62], q[17];
  cx q[76], q[31];
  cx q[90], q[45];
  cx q[55], q[11];
  cx q[69], q[25];
  cx q[83], q[39];
  cx q[51], q[5];
  cx q[64], q[19];
  cx q[78], q[33];
  cx q[92], q[47];
  cx q[57], q[13];
  cx q[71], q[27];
  cx q[85], q[41];
  cx q[14], q[59];
  cx q[28], q[73];
  cx q[42], q[87];
  cx q[8], q[52];
  cx q[22], q[66];
  cx q[36], q[80];
  cx q[16], q[61];
  cx q[30], q[75];
  cx q[44], q[89];
  cx q[10], q[54];
  cx q[24], q[68];
  cx q[38], q[82];
  cx q[18], q[63];
  cx q[32], q[77];
  cx q[46], q[91];
  cx q[12], q[56];
  cx q[26], q[70];
  cx q[40], q[84];
  cx q[20], q[65];
  cx q[34], q[79];
  cx q[48], q[93];

  // CX step 2
  cx q[49], q[0];
  cx q[60], q[14];
  cx q[74], q[28];
  cx q[88], q[42];
  cx q[53], q[8];
  cx q[67], q[22];
  cx q[81], q[36];
  cx q[50], q[2];
  cx q[62], q[16];
  cx q[76], q[30];
  cx q[90], q[44];
  cx q[55], q[10];
  cx q[69], q[24];
  cx q[83], q[38];
  cx q[51], q[4];
  cx q[64], q[18];
  cx q[78], q[32];
  cx q[92], q[46];
  cx q[57], q[12];
  cx q[71], q[26];
  cx q[85], q[40];
  cx q[7], q[59];
  cx q[21], q[73];
  cx q[35], q[87];
  cx q[1], q[52];
  cx q[15], q[66];
  cx q[29], q[80];
  cx q[9], q[61];
  cx q[23], q[75];
  cx q[37], q[89];
  cx q[3], q[54];
  cx q[17], q[68];
  cx q[31], q[82];
  cx q[11], q[63];
  cx q[25], q[77];
  cx q[39], q[91];
  cx q[5], q[56];
  cx q[19], q[70];
  cx q[33], q[84];
  cx q[13], q[65];
  cx q[27], q[79];
  cx q[41], q[93];

  // CX step 3
  cx q[60], q[8];
  cx q[74], q[22];
  cx q[88], q[36];
  cx q[53], q[2];
  cx q[67], q[16];
  cx q[81], q[30];
  cx q[94], q[44];
  cx q[62], q[10];
  cx q[76], q[24];
  cx q[90], q[38];
  cx q[55], q[4];
  cx q[69], q[18];
  cx q[83], q[32];
  cx q[95], q[46];
  cx q[64], q[12];
  cx q[78], q[26];
  cx q[92], q[40];
  cx q[57], q[6];
  cx q[71], q[20];
  cx q[85], q[34];
  cx q[96], q[48];
  cx q[7], q[52];
  cx q[21], q[66];
  cx q[35], q[80];
  cx q[15], q[61];
  cx q[29], q[75];
  cx q[43], q[89];
  cx q[9], q[54];
  cx q[23], q[68];
  cx q[37], q[82];
  cx q[17], q[63];
  cx q[31], q[77];
  cx q[45], q[91];
  cx q[11], q[56];
  cx q[25], q[70];
  cx q[39], q[84];
  cx q[19], q[65];
  cx q[33], q[79];
  cx q[47], q[93];
  cx q[13], q[58];
  cx q[27], q[72];
  cx q[41], q[86];

  // CX step 4
  cx q[60], q[7];
  cx q[74], q[21];
  cx q[88], q[35];
  cx q[53], q[1];
  cx q[67], q[15];
  cx q[81], q[29];
  cx q[94], q[43];
  cx q[62], q[9];
  cx q[76], q[23];
  cx q[90], q[37];
  cx q[55], q[3];
  cx q[69], q[17];
  cx q[83], q[31];
  cx q[95], q[45];
  cx q[64], q[11];
  cx q[78], q[25];
  cx q[92], q[39];
  cx q[57], q[5];
  cx q[71], q[19];
  cx q[85], q[33];
  cx q[96], q[47];
  cx q[0], q[52];
  cx q[14], q[66];
  cx q[28], q[80];
  cx q[8], q[61];
  cx q[22], q[75];
  cx q[36], q[89];
  cx q[2], q[54];
  cx q[16], q[68];
  cx q[30], q[82];
  cx q[10], q[63];
  cx q[24], q[77];
  cx q[38], q[91];
  cx q[4], q[56];
  cx q[18], q[70];
  cx q[32], q[84];
  cx q[12], q[65];
  cx q[26], q[79];
  cx q[40], q[93];
  cx q[6], q[58];
  cx q[20], q[72];
  cx q[34], q[86];

  // Undo Hadamard on X-stabilizer ancillas
  h q[49];
  h q[50];
  h q[51];
  h q[53];
  h q[55];
  h q[57];
  h q[60];
  h q[62];
  h q[64];
  h q[67];
  h q[69];
  h q[71];
  h q[74];
  h q[76];
  h q[78];
  h q[81];
  h q[83];
  h q[85];
  h q[88];
  h q[90];
  h q[92];
  h q[94];
  h q[95];
  h q[96];

  // Measure syndrome ancillas
  c[0] = measure q[52];
  c[1] = measure q[54];
  c[2] = measure q[56];
  c[3] = measure q[58];
  c[4] = measure q[59];
  c[5] = measure q[61];
  c[6] = measure q[63];
  c[7] = measure q[65];
  c[8] = measure q[66];
  c[9] = measure q[68];
  c[10] = measure q[70];
  c[11] = measure q[72];
  c[12] = measure q[73];
  c[13] = measure q[75];
  c[14] = measure q[77];
  c[15] = measure q[79];
  c[16] = measure q[80];
  c[17] = measure q[82];
  c[18] = measure q[84];
  c[19] = measure q[86];
  c[20] = measure q[87];
  c[21] = measure q[89];
  c[22] = measure q[91];
  c[23] = measure q[93];
  c[24] = measure q[49];
  c[25] = measure q[50];
  c[26] = measure q[51];
  c[27] = measure q[53];
  c[28] = measure q[55];
  c[29] = measure q[57];
  c[30] = measure q[60];
  c[31] = measure q[62];
  c[32] = measure q[64];
  c[33] = measure q[67];
  c[34] = measure q[69];
  c[35] = measure q[71];
  c[36] = measure q[74];
  c[37] = measure q[76];
  c[38] = measure q[78];
  c[39] = measure q[81];
  c[40] = measure q[83];
  c[41] = measure q[85];
  c[42] = measure q[88];
  c[43] = measure q[90];
  c[44] = measure q[92];
  c[45] = measure q[94];
  c[46] = measure q[95];
  c[47] = measure q[96];

  // Conditional corrections
  if (c[0]) {
    x q[8];
  }
  if (c[1]) {
    x q[10];
  }
  if (c[2]) {
    x q[12];
  }
  if (c[3]) {
    x q[13];
  }
  if (c[4]) {
    x q[14];
  }
  if (c[5]) {
    x q[16];
  }
  if (c[6]) {
    x q[18];
  }
  if (c[7]) {
    x q[20];
  }
  if (c[8]) {
    x q[22];
  }
  if (c[9]) {
    x q[24];
  }
  if (c[10]) {
    x q[26];
  }
  if (c[11]) {
    x q[27];
  }
  if (c[12]) {
    x q[28];
  }
  if (c[13]) {
    x q[30];
  }
  if (c[14]) {
    x q[32];
  }
  if (c[15]) {
    x q[34];
  }
  if (c[16]) {
    x q[36];
  }
  if (c[17]) {
    x q[38];
  }
  if (c[18]) {
    x q[40];
  }
  if (c[19]) {
    x q[41];
  }
  if (c[20]) {
    x q[42];
  }
  if (c[21]) {
    x q[44];
  }
  if (c[22]) {
    x q[46];
  }
  if (c[23]) {
    x q[48];
  }
  if (c[24]) {
    z q[1];
  }
  if (c[25]) {
    z q[3];
  }
  if (c[26]) {
    z q[5];
  }
  if (c[27]) {
    z q[9];
  }
  if (c[28]) {
    z q[11];
  }
  if (c[29]) {
    z q[13];
  }
  if (c[30]) {
    z q[15];
  }
  if (c[31]) {
    z q[17];
  }
  if (c[32]) {
    z q[19];
  }
  if (c[33]) {
    z q[23];
  }
  if (c[34]) {
    z q[25];
  }
  if (c[35]) {
    z q[27];
  }
  if (c[36]) {
    z q[29];
  }
  if (c[37]) {
    z q[31];
  }
  if (c[38]) {
    z q[33];
  }
  if (c[39]) {
    z q[37];
  }
  if (c[40]) {
    z q[39];
  }
  if (c[41]) {
    z q[41];
  }
  if (c[42]) {
    z q[43];
  }
  if (c[43]) {
    z q[45];
  }
  if (c[44]) {
    z q[47];
  }
  if (c[45]) {
    z q[44];
  }
  if (c[46]) {
    z q[46];
  }
  if (c[47]) {
    z q[48];
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
c[24] = measure q[24];
c[25] = measure q[25];
c[26] = measure q[26];
c[27] = measure q[27];
c[28] = measure q[28];
c[29] = measure q[29];
c[30] = measure q[30];
c[31] = measure q[31];
c[32] = measure q[32];
c[33] = measure q[33];
c[34] = measure q[34];
c[35] = measure q[35];
c[36] = measure q[36];
c[37] = measure q[37];
c[38] = measure q[38];
c[39] = measure q[39];
c[40] = measure q[40];
c[41] = measure q[41];
c[42] = measure q[42];
c[43] = measure q[43];
c[44] = measure q[44];
c[45] = measure q[45];
c[46] = measure q[46];
c[47] = measure q[47];