OPENQASM 3;
include "stdgates.inc";
qubit[97] q;
bit[48] c;

// Surface Code d=7, 3 syndrome extraction rounds
// 49 data qubits + 48 ancilla qubits = 97 total
// X-stabilizers: 24, Z-stabilizers: 24

h q[0];
h q[4];
h q[7];
h q[9];
h q[11];
h q[13];
h q[15];
h q[21];
h q[25];
h q[34];
h q[35];
h q[36];
h q[38];
h q[43];
h q[44];
h q[45];

for int round in [0:3] {
  // Reset ancilla qubits
  reset q[50];
  reset q[52];
  reset q[54];
  reset q[55];
  reset q[57];
  reset q[59];
  reset q[62];
  reset q[64];
  reset q[66];
  reset q[67];
  reset q[69];
  reset q[71];
  reset q[74];
  reset q[76];
  reset q[78];
  reset q[79];
  reset q[81];
  reset q[83];
  reset q[85];
  reset q[86];
  reset q[87];
  reset q[88];
  reset q[89];
  reset q[90];
  reset q[49];
  reset q[51];
  reset q[53];
  reset q[56];
  reset q[58];
  reset q[60];
  reset q[61];
  reset q[63];
  reset q[65];
  reset q[68];
  reset q[70];
  reset q[72];
  reset q[73];
  reset q[75];
  reset q[77];
  reset q[80];
  reset q[82];
  reset q[84];
  reset q[91];
  reset q[92];
  reset q[93];
  reset q[94];
  reset q[95];
  reset q[96];

  // Z-stabilizer syndrome extraction
  cx q[1], q[50];
  cx q[2], q[50];
  cx q[8], q[50];
  cx q[9], q[50];
  cx q[3], q[52];
  cx q[4], q[52];
  cx q[10], q[52];
  cx q[11], q[52];
  cx q[5], q[54];
  cx q[6], q[54];
  cx q[12], q[54];
  cx q[13], q[54];
  cx q[7], q[55];
  cx q[8], q[55];
  cx q[14], q[55];
  cx q[15], q[55];
  cx q[9], q[57];
  cx q[10], q[57];
  cx q[16], q[57];
  cx q[17], q[57];
  cx q[11], q[59];
  cx q[12], q[59];
  cx q[18], q[59];
  cx q[19], q[59];
  cx q[15], q[62];
  cx q[16], q[62];
  cx q[22], q[62];
  cx q[23], q[62];
  cx q[17], q[64];
  cx q[18], q[64];
  cx q[24], q[64];
  cx q[25], q[64];
  cx q[19], q[66];
  cx q[20], q[66];
  cx q[26], q[66];
  cx q[27], q[66];
  cx q[21], q[67];
  cx q[22], q[67];
  cx q[28], q[67];
  cx q[29], q[67];
  cx q[23], q[69];
  cx q[24], q[69];
  cx q[30], q[69];
  cx q[31], q[69];
  cx q[25], q[71];
  cx q[26], q[71];
  cx q[32], q[71];
  cx q[33], q[71];
  cx q[29], q[74];
  cx q[30], q[74];
  cx q[36], q[74];
  cx q[37], q[74];
  cx q[31], q[76];
  cx q[32], q[76];
  cx q[38], q[76];
  cx q[39], q[76];
  cx q[33], q[78];
  cx q[34], q[78];
  cx q[40], q[78];
  cx q[41], q[78];
  cx q[35], q[79];
  cx q[36], q[79];
  cx q[42], q[79];
  cx q[43], q[79];
  cx q[37], q[81];
  cx q[38], q[81];
  cx q[44], q[81];
  cx q[45], q[81];
  cx q[39], q[83];
  cx q[40], q[83];
  cx q[46], q[83];
  cx q[47], q[83];
  cx q[0], q[85];
  cx q[1], q[85];
  cx q[2], q[86];
  cx q[3], q[86];
  cx q[4], q[87];
  cx q[5], q[87];
  cx q[42], q[88];
  cx q[43], q[88];
  cx q[44], q[89];
  cx q[45], q[89];
  cx q[46], q[90];
  cx q[47], q[90];

  // X-stabilizer syndrome extraction
  h q[49];
  cx q[49], q[0];
  cx q[49], q[1];
  cx q[49], q[7];
  cx q[49], q[8];
  h q[49];
  h q[51];
  cx q[51], q[2];
  cx q[51], q[3];
  cx q[51], q[9];
  cx q[51], q[10];
  h q[51];
  h q[53];
  cx q[53], q[4];
  cx q[53], q[5];
  cx q[53], q[11];
  cx q[53], q[12];
  h q[53];
  h q[56];
  cx q[56], q[8];
  cx q[56], q[9];
  cx q[56], q[15];
  cx q[56], q[16];
  h q[56];
  h q[58];
  cx q[58], q[10];
  cx q[58], q[11];
  cx q[58], q[17];
  cx q[58], q[18];
  h q[58];
  h q[60];
  cx q[60], q[12];
  cx q[60], q[13];
  cx q[60], q[19];
  cx q[60], q[20];
  h q[60];
  h q[61];
  cx q[61], q[14];
  cx q[61], q[15];
  cx q[61], q[21];
  cx q[61], q[22];
  h q[61];
  h q[63];
  cx q[63], q[16];
  cx q[63], q[17];
  cx q[63], q[23];
  cx q[63], q[24];
  h q[63];
  h q[65];
  cx q[65], q[18];
  cx q[65], q[19];
  cx q[65], q[25];
  cx q[65], q[26];
  h q[65];
  h q[68];
  cx q[68], q[22];
  cx q[68], q[23];
  cx q[68], q[29];
  cx q[68], q[30];
  h q[68];
  h q[70];
  cx q[70], q[24];
  cx q[70], q[25];
  cx q[70], q[31];
  cx q[70], q[32];
  h q[70];
  h q[72];
  cx q[72], q[26];
  cx q[72], q[27];
  cx q[72], q[33];
  cx q[72], q[34];
  h q[72];
  h q[73];
  cx q[73], q[28];
  cx q[73], q[29];
  cx q[73], q[35];
  cx q[73], q[36];
  h q[73];
  h q[75];
  cx q[75], q[30];
  cx q[75], q[31];
  cx q[75], q[37];
  cx q[75], q[38];
  h q[75];
  h q[77];
  cx q[77], q[32];
  cx q[77], q[33];
  cx q[77], q[39];
  cx q[77], q[40];
  h q[77];
  h q[80];
  cx q[80], q[36];
  cx q[80], q[37];
  cx q[80], q[43];
  cx q[80], q[44];
  h q[80];
  h q[82];
  cx q[82], q[38];
  cx q[82], q[39];
  cx q[82], q[45];
  cx q[82], q[46];
  h q[82];
  h q[84];
  cx q[84], q[40];
  cx q[84], q[41];
  cx q[84], q[47];
  cx q[84], q[48];
  h q[84];
  h q[91];
  cx q[91], q[0];
  cx q[91], q[7];
  h q[91];
  h q[92];
  cx q[92], q[14];
  cx q[92], q[21];
  h q[92];
  h q[93];
  cx q[93], q[28];
  cx q[93], q[35];
  h q[93];
  h q[94];
  cx q[94], q[6];
  cx q[94], q[13];
  h q[94];
  h q[95];
  cx q[95], q[20];
  cx q[95], q[27];
  h q[95];
  h q[96];
  cx q[96], q[34];
  cx q[96], q[41];
  h q[96];

  // Measure syndrome ancillas
  c[0] = measure q[50];
  c[1] = measure q[52];
  c[2] = measure q[54];
  c[3] = measure q[55];
  c[4] = measure q[57];
  c[5] = measure q[59];
  c[6] = measure q[62];
  c[7] = measure q[64];
  c[8] = measure q[66];
  c[9] = measure q[67];
  c[10] = measure q[69];
  c[11] = measure q[71];
  c[12] = measure q[74];
  c[13] = measure q[76];
  c[14] = measure q[78];
  c[15] = measure q[79];
  c[16] = measure q[81];
  c[17] = measure q[83];
  c[18] = measure q[85];
  c[19] = measure q[86];
  c[20] = measure q[87];
  c[21] = measure q[88];
  c[22] = measure q[89];
  c[23] = measure q[90];
  c[24] = measure q[49];
  c[25] = measure q[51];
  c[26] = measure q[53];
  c[27] = measure q[56];
  c[28] = measure q[58];
  c[29] = measure q[60];
  c[30] = measure q[61];
  c[31] = measure q[63];
  c[32] = measure q[65];
  c[33] = measure q[68];
  c[34] = measure q[70];
  c[35] = measure q[72];
  c[36] = measure q[73];
  c[37] = measure q[75];
  c[38] = measure q[77];
  c[39] = measure q[80];
  c[40] = measure q[82];
  c[41] = measure q[84];
  c[42] = measure q[91];
  c[43] = measure q[92];
  c[44] = measure q[93];
  c[45] = measure q[94];
  c[46] = measure q[95];
  c[47] = measure q[96];

  // Conditional corrections
  if (c[0]) {
    x q[1];
  }
  if (c[1]) {
    x q[3];
  }
  if (c[2]) {
    x q[5];
  }
  if (c[3]) {
    x q[7];
  }
  if (c[4]) {
    x q[9];
  }
  if (c[5]) {
    x q[11];
  }
  if (c[6]) {
    x q[15];
  }
  if (c[7]) {
    x q[17];
  }
  if (c[8]) {
    x q[19];
  }
  if (c[9]) {
    x q[21];
  }
  if (c[10]) {
    x q[23];
  }
  if (c[11]) {
    x q[25];
  }
  if (c[12]) {
    x q[29];
  }
  if (c[13]) {
    x q[31];
  }
  if (c[14]) {
    x q[33];
  }
  if (c[15]) {
    x q[35];
  }
  if (c[16]) {
    x q[37];
  }
  if (c[17]) {
    x q[39];
  }
  if (c[18]) {
    x q[0];
  }
  if (c[19]) {
    x q[2];
  }
  if (c[20]) {
    x q[4];
  }
  if (c[21]) {
    x q[42];
  }
  if (c[22]) {
    x q[44];
  }
  if (c[23]) {
    x q[46];
  }
  if (c[24]) {
    z q[0];
  }
  if (c[25]) {
    z q[2];
  }
  if (c[26]) {
    z q[4];
  }
  if (c[27]) {
    z q[8];
  }
  if (c[28]) {
    z q[10];
  }
  if (c[29]) {
    z q[12];
  }
  if (c[30]) {
    z q[14];
  }
  if (c[31]) {
    z q[16];
  }
  if (c[32]) {
    z q[18];
  }
  if (c[33]) {
    z q[22];
  }
  if (c[34]) {
    z q[24];
  }
  if (c[35]) {
    z q[26];
  }
  if (c[36]) {
    z q[28];
  }
  if (c[37]) {
    z q[30];
  }
  if (c[38]) {
    z q[32];
  }
  if (c[39]) {
    z q[36];
  }
  if (c[40]) {
    z q[38];
  }
  if (c[41]) {
    z q[40];
  }
  if (c[42]) {
    z q[0];
  }
  if (c[43]) {
    z q[14];
  }
  if (c[44]) {
    z q[28];
  }
  if (c[45]) {
    z q[6];
  }
  if (c[46]) {
    z q[20];
  }
  if (c[47]) {
    z q[34];
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