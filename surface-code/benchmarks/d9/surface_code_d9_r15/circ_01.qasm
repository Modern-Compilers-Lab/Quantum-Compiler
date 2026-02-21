OPENQASM 3;
include "stdgates.inc";
qubit[161] q;
bit[80] c;

// Surface Code d=9, 15 syndrome extraction rounds
// 81 data qubits + 80 ancilla qubits = 161 total
// X-stabilizers: 40, Z-stabilizers: 40

h q[2];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[17];
h q[20];
h q[27];
h q[29];
h q[32];
h q[37];
h q[41];
h q[44];
h q[45];
h q[47];
h q[49];
h q[55];
h q[57];
h q[61];
h q[64];
h q[67];
h q[68];
h q[69];
h q[71];
h q[73];
h q[77];

for int round in [0:15] {
  // Reset ancilla qubits
  reset q[82];
  reset q[84];
  reset q[86];
  reset q[88];
  reset q[89];
  reset q[91];
  reset q[93];
  reset q[95];
  reset q[98];
  reset q[100];
  reset q[102];
  reset q[104];
  reset q[105];
  reset q[107];
  reset q[109];
  reset q[111];
  reset q[114];
  reset q[116];
  reset q[118];
  reset q[120];
  reset q[121];
  reset q[123];
  reset q[125];
  reset q[127];
  reset q[130];
  reset q[132];
  reset q[134];
  reset q[136];
  reset q[137];
  reset q[139];
  reset q[141];
  reset q[143];
  reset q[145];
  reset q[146];
  reset q[147];
  reset q[148];
  reset q[149];
  reset q[150];
  reset q[151];
  reset q[152];
  reset q[81];
  reset q[83];
  reset q[85];
  reset q[87];
  reset q[90];
  reset q[92];
  reset q[94];
  reset q[96];
  reset q[97];
  reset q[99];
  reset q[101];
  reset q[103];
  reset q[106];
  reset q[108];
  reset q[110];
  reset q[112];
  reset q[113];
  reset q[115];
  reset q[117];
  reset q[119];
  reset q[122];
  reset q[124];
  reset q[126];
  reset q[128];
  reset q[129];
  reset q[131];
  reset q[133];
  reset q[135];
  reset q[138];
  reset q[140];
  reset q[142];
  reset q[144];
  reset q[153];
  reset q[154];
  reset q[155];
  reset q[156];
  reset q[157];
  reset q[158];
  reset q[159];
  reset q[160];

  // Z-stabilizer syndrome extraction
  cx q[1], q[82];
  cx q[2], q[82];
  cx q[10], q[82];
  cx q[11], q[82];
  cx q[3], q[84];
  cx q[4], q[84];
  cx q[12], q[84];
  cx q[13], q[84];
  cx q[5], q[86];
  cx q[6], q[86];
  cx q[14], q[86];
  cx q[15], q[86];
  cx q[7], q[88];
  cx q[8], q[88];
  cx q[16], q[88];
  cx q[17], q[88];
  cx q[9], q[89];
  cx q[10], q[89];
  cx q[18], q[89];
  cx q[19], q[89];
  cx q[11], q[91];
  cx q[12], q[91];
  cx q[20], q[91];
  cx q[21], q[91];
  cx q[13], q[93];
  cx q[14], q[93];
  cx q[22], q[93];
  cx q[23], q[93];
  cx q[15], q[95];
  cx q[16], q[95];
  cx q[24], q[95];
  cx q[25], q[95];
  cx q[19], q[98];
  cx q[20], q[98];
  cx q[28], q[98];
  cx q[29], q[98];
  cx q[21], q[100];
  cx q[22], q[100];
  cx q[30], q[100];
  cx q[31], q[100];
  cx q[23], q[102];
  cx q[24], q[102];
  cx q[32], q[102];
  cx q[33], q[102];
  cx q[25], q[104];
  cx q[26], q[104];
  cx q[34], q[104];
  cx q[35], q[104];
  cx q[27], q[105];
  cx q[28], q[105];
  cx q[36], q[105];
  cx q[37], q[105];
  cx q[29], q[107];
  cx q[30], q[107];
  cx q[38], q[107];
  cx q[39], q[107];
  cx q[31], q[109];
  cx q[32], q[109];
  cx q[40], q[109];
  cx q[41], q[109];
  cx q[33], q[111];
  cx q[34], q[111];
  cx q[42], q[111];
  cx q[43], q[111];
  cx q[37], q[114];
  cx q[38], q[114];
  cx q[46], q[114];
  cx q[47], q[114];
  cx q[39], q[116];
  cx q[40], q[116];
  cx q[48], q[116];
  cx q[49], q[116];
  cx q[41], q[118];
  cx q[42], q[118];
  cx q[50], q[118];
  cx q[51], q[118];
  cx q[43], q[120];
  cx q[44], q[120];
  cx q[52], q[120];
  cx q[53], q[120];
  cx q[45], q[121];
  cx q[46], q[121];
  cx q[54], q[121];
  cx q[55], q[121];
  cx q[47], q[123];
  cx q[48], q[123];
  cx q[56], q[123];
  cx q[57], q[123];
  cx q[49], q[125];
  cx q[50], q[125];
  cx q[58], q[125];
  cx q[59], q[125];
  cx q[51], q[127];
  cx q[52], q[127];
  cx q[60], q[127];
  cx q[61], q[127];
  cx q[55], q[130];
  cx q[56], q[130];
  cx q[64], q[130];
  cx q[65], q[130];
  cx q[57], q[132];
  cx q[58], q[132];
  cx q[66], q[132];
  cx q[67], q[132];
  cx q[59], q[134];
  cx q[60], q[134];
  cx q[68], q[134];
  cx q[69], q[134];
  cx q[61], q[136];
  cx q[62], q[136];
  cx q[70], q[136];
  cx q[71], q[136];
  cx q[63], q[137];
  cx q[64], q[137];
  cx q[72], q[137];
  cx q[73], q[137];
  cx q[65], q[139];
  cx q[66], q[139];
  cx q[74], q[139];
  cx q[75], q[139];
  cx q[67], q[141];
  cx q[68], q[141];
  cx q[76], q[141];
  cx q[77], q[141];
  cx q[69], q[143];
  cx q[70], q[143];
  cx q[78], q[143];
  cx q[79], q[143];
  cx q[0], q[145];
  cx q[1], q[145];
  cx q[2], q[146];
  cx q[3], q[146];
  cx q[4], q[147];
  cx q[5], q[147];
  cx q[6], q[148];
  cx q[7], q[148];
  cx q[72], q[149];
  cx q[73], q[149];
  cx q[74], q[150];
  cx q[75], q[150];
  cx q[76], q[151];
  cx q[77], q[151];
  cx q[78], q[152];
  cx q[79], q[152];

  // X-stabilizer syndrome extraction
  h q[81];
  cx q[81], q[0];
  cx q[81], q[1];
  cx q[81], q[9];
  cx q[81], q[10];
  h q[81];
  h q[83];
  cx q[83], q[2];
  cx q[83], q[3];
  cx q[83], q[11];
  cx q[83], q[12];
  h q[83];
  h q[85];
  cx q[85], q[4];
  cx q[85], q[5];
  cx q[85], q[13];
  cx q[85], q[14];
  h q[85];
  h q[87];
  cx q[87], q[6];
  cx q[87], q[7];
  cx q[87], q[15];
  cx q[87], q[16];
  h q[87];
  h q[90];
  cx q[90], q[10];
  cx q[90], q[11];
  cx q[90], q[19];
  cx q[90], q[20];
  h q[90];
  h q[92];
  cx q[92], q[12];
  cx q[92], q[13];
  cx q[92], q[21];
  cx q[92], q[22];
  h q[92];
  h q[94];
  cx q[94], q[14];
  cx q[94], q[15];
  cx q[94], q[23];
  cx q[94], q[24];
  h q[94];
  h q[96];
  cx q[96], q[16];
  cx q[96], q[17];
  cx q[96], q[25];
  cx q[96], q[26];
  h q[96];
  h q[97];
  cx q[97], q[18];
  cx q[97], q[19];
  cx q[97], q[27];
  cx q[97], q[28];
  h q[97];
  h q[99];
  cx q[99], q[20];
  cx q[99], q[21];
  cx q[99], q[29];
  cx q[99], q[30];
  h q[99];
  h q[101];
  cx q[101], q[22];
  cx q[101], q[23];
  cx q[101], q[31];
  cx q[101], q[32];
  h q[101];
  h q[103];
  cx q[103], q[24];
  cx q[103], q[25];
  cx q[103], q[33];
  cx q[103], q[34];
  h q[103];
  h q[106];
  cx q[106], q[28];
  cx q[106], q[29];
  cx q[106], q[37];
  cx q[106], q[38];
  h q[106];
  h q[108];
  cx q[108], q[30];
  cx q[108], q[31];
  cx q[108], q[39];
  cx q[108], q[40];
  h q[108];
  h q[110];
  cx q[110], q[32];
  cx q[110], q[33];
  cx q[110], q[41];
  cx q[110], q[42];
  h q[110];
  h q[112];
  cx q[112], q[34];
  cx q[112], q[35];
  cx q[112], q[43];
  cx q[112], q[44];
  h q[112];
  h q[113];
  cx q[113], q[36];
  cx q[113], q[37];
  cx q[113], q[45];
  cx q[113], q[46];
  h q[113];
  h q[115];
  cx q[115], q[38];
  cx q[115], q[39];
  cx q[115], q[47];
  cx q[115], q[48];
  h q[115];
  h q[117];
  cx q[117], q[40];
  cx q[117], q[41];
  cx q[117], q[49];
  cx q[117], q[50];
  h q[117];
  h q[119];
  cx q[119], q[42];
  cx q[119], q[43];
  cx q[119], q[51];
  cx q[119], q[52];
  h q[119];
  h q[122];
  cx q[122], q[46];
  cx q[122], q[47];
  cx q[122], q[55];
  cx q[122], q[56];
  h q[122];
  h q[124];
  cx q[124], q[48];
  cx q[124], q[49];
  cx q[124], q[57];
  cx q[124], q[58];
  h q[124];
  h q[126];
  cx q[126], q[50];
  cx q[126], q[51];
  cx q[126], q[59];
  cx q[126], q[60];
  h q[126];
  h q[128];
  cx q[128], q[52];
  cx q[128], q[53];
  cx q[128], q[61];
  cx q[128], q[62];
  h q[128];
  h q[129];
  cx q[129], q[54];
  cx q[129], q[55];
  cx q[129], q[63];
  cx q[129], q[64];
  h q[129];
  h q[131];
  cx q[131], q[56];
  cx q[131], q[57];
  cx q[131], q[65];
  cx q[131], q[66];
  h q[131];
  h q[133];
  cx q[133], q[58];
  cx q[133], q[59];
  cx q[133], q[67];
  cx q[133], q[68];
  h q[133];
  h q[135];
  cx q[135], q[60];
  cx q[135], q[61];
  cx q[135], q[69];
  cx q[135], q[70];
  h q[135];
  h q[138];
  cx q[138], q[64];
  cx q[138], q[65];
  cx q[138], q[73];
  cx q[138], q[74];
  h q[138];
  h q[140];
  cx q[140], q[66];
  cx q[140], q[67];
  cx q[140], q[75];
  cx q[140], q[76];
  h q[140];
  h q[142];
  cx q[142], q[68];
  cx q[142], q[69];
  cx q[142], q[77];
  cx q[142], q[78];
  h q[142];
  h q[144];
  cx q[144], q[70];
  cx q[144], q[71];
  cx q[144], q[79];
  cx q[144], q[80];
  h q[144];
  h q[153];
  cx q[153], q[0];
  cx q[153], q[9];
  h q[153];
  h q[154];
  cx q[154], q[18];
  cx q[154], q[27];
  h q[154];
  h q[155];
  cx q[155], q[36];
  cx q[155], q[45];
  h q[155];
  h q[156];
  cx q[156], q[54];
  cx q[156], q[63];
  h q[156];
  h q[157];
  cx q[157], q[8];
  cx q[157], q[17];
  h q[157];
  h q[158];
  cx q[158], q[26];
  cx q[158], q[35];
  h q[158];
  h q[159];
  cx q[159], q[44];
  cx q[159], q[53];
  h q[159];
  h q[160];
  cx q[160], q[62];
  cx q[160], q[71];
  h q[160];

  // Measure syndrome ancillas
  c[0] = measure q[82];
  c[1] = measure q[84];
  c[2] = measure q[86];
  c[3] = measure q[88];
  c[4] = measure q[89];
  c[5] = measure q[91];
  c[6] = measure q[93];
  c[7] = measure q[95];
  c[8] = measure q[98];
  c[9] = measure q[100];
  c[10] = measure q[102];
  c[11] = measure q[104];
  c[12] = measure q[105];
  c[13] = measure q[107];
  c[14] = measure q[109];
  c[15] = measure q[111];
  c[16] = measure q[114];
  c[17] = measure q[116];
  c[18] = measure q[118];
  c[19] = measure q[120];
  c[20] = measure q[121];
  c[21] = measure q[123];
  c[22] = measure q[125];
  c[23] = measure q[127];
  c[24] = measure q[130];
  c[25] = measure q[132];
  c[26] = measure q[134];
  c[27] = measure q[136];
  c[28] = measure q[137];
  c[29] = measure q[139];
  c[30] = measure q[141];
  c[31] = measure q[143];
  c[32] = measure q[145];
  c[33] = measure q[146];
  c[34] = measure q[147];
  c[35] = measure q[148];
  c[36] = measure q[149];
  c[37] = measure q[150];
  c[38] = measure q[151];
  c[39] = measure q[152];
  c[40] = measure q[81];
  c[41] = measure q[83];
  c[42] = measure q[85];
  c[43] = measure q[87];
  c[44] = measure q[90];
  c[45] = measure q[92];
  c[46] = measure q[94];
  c[47] = measure q[96];
  c[48] = measure q[97];
  c[49] = measure q[99];
  c[50] = measure q[101];
  c[51] = measure q[103];
  c[52] = measure q[106];
  c[53] = measure q[108];
  c[54] = measure q[110];
  c[55] = measure q[112];
  c[56] = measure q[113];
  c[57] = measure q[115];
  c[58] = measure q[117];
  c[59] = measure q[119];
  c[60] = measure q[122];
  c[61] = measure q[124];
  c[62] = measure q[126];
  c[63] = measure q[128];
  c[64] = measure q[129];
  c[65] = measure q[131];
  c[66] = measure q[133];
  c[67] = measure q[135];
  c[68] = measure q[138];
  c[69] = measure q[140];
  c[70] = measure q[142];
  c[71] = measure q[144];
  c[72] = measure q[153];
  c[73] = measure q[154];
  c[74] = measure q[155];
  c[75] = measure q[156];
  c[76] = measure q[157];
  c[77] = measure q[158];
  c[78] = measure q[159];
  c[79] = measure q[160];

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
    x q[13];
  }
  if (c[7]) {
    x q[15];
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
    x q[27];
  }
  if (c[13]) {
    x q[29];
  }
  if (c[14]) {
    x q[31];
  }
  if (c[15]) {
    x q[33];
  }
  if (c[16]) {
    x q[37];
  }
  if (c[17]) {
    x q[39];
  }
  if (c[18]) {
    x q[41];
  }
  if (c[19]) {
    x q[43];
  }
  if (c[20]) {
    x q[45];
  }
  if (c[21]) {
    x q[47];
  }
  if (c[22]) {
    x q[49];
  }
  if (c[23]) {
    x q[51];
  }
  if (c[24]) {
    x q[55];
  }
  if (c[25]) {
    x q[57];
  }
  if (c[26]) {
    x q[59];
  }
  if (c[27]) {
    x q[61];
  }
  if (c[28]) {
    x q[63];
  }
  if (c[29]) {
    x q[65];
  }
  if (c[30]) {
    x q[67];
  }
  if (c[31]) {
    x q[69];
  }
  if (c[32]) {
    x q[0];
  }
  if (c[33]) {
    x q[2];
  }
  if (c[34]) {
    x q[4];
  }
  if (c[35]) {
    x q[6];
  }
  if (c[36]) {
    x q[72];
  }
  if (c[37]) {
    x q[74];
  }
  if (c[38]) {
    x q[76];
  }
  if (c[39]) {
    x q[78];
  }
  if (c[40]) {
    z q[0];
  }
  if (c[41]) {
    z q[2];
  }
  if (c[42]) {
    z q[4];
  }
  if (c[43]) {
    z q[6];
  }
  if (c[44]) {
    z q[10];
  }
  if (c[45]) {
    z q[12];
  }
  if (c[46]) {
    z q[14];
  }
  if (c[47]) {
    z q[16];
  }
  if (c[48]) {
    z q[18];
  }
  if (c[49]) {
    z q[20];
  }
  if (c[50]) {
    z q[22];
  }
  if (c[51]) {
    z q[24];
  }
  if (c[52]) {
    z q[28];
  }
  if (c[53]) {
    z q[30];
  }
  if (c[54]) {
    z q[32];
  }
  if (c[55]) {
    z q[34];
  }
  if (c[56]) {
    z q[36];
  }
  if (c[57]) {
    z q[38];
  }
  if (c[58]) {
    z q[40];
  }
  if (c[59]) {
    z q[42];
  }
  if (c[60]) {
    z q[46];
  }
  if (c[61]) {
    z q[48];
  }
  if (c[62]) {
    z q[50];
  }
  if (c[63]) {
    z q[52];
  }
  if (c[64]) {
    z q[54];
  }
  if (c[65]) {
    z q[56];
  }
  if (c[66]) {
    z q[58];
  }
  if (c[67]) {
    z q[60];
  }
  if (c[68]) {
    z q[64];
  }
  if (c[69]) {
    z q[66];
  }
  if (c[70]) {
    z q[68];
  }
  if (c[71]) {
    z q[70];
  }
  if (c[72]) {
    z q[0];
  }
  if (c[73]) {
    z q[18];
  }
  if (c[74]) {
    z q[36];
  }
  if (c[75]) {
    z q[54];
  }
  if (c[76]) {
    z q[8];
  }
  if (c[77]) {
    z q[26];
  }
  if (c[78]) {
    z q[44];
  }
  if (c[79]) {
    z q[62];
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
c[48] = measure q[48];
c[49] = measure q[49];
c[50] = measure q[50];
c[51] = measure q[51];
c[52] = measure q[52];
c[53] = measure q[53];
c[54] = measure q[54];
c[55] = measure q[55];
c[56] = measure q[56];
c[57] = measure q[57];
c[58] = measure q[58];
c[59] = measure q[59];
c[60] = measure q[60];
c[61] = measure q[61];
c[62] = measure q[62];
c[63] = measure q[63];
c[64] = measure q[64];
c[65] = measure q[65];
c[66] = measure q[66];
c[67] = measure q[67];
c[68] = measure q[68];
c[69] = measure q[69];
c[70] = measure q[70];
c[71] = measure q[71];
c[72] = measure q[72];
c[73] = measure q[73];
c[74] = measure q[74];
c[75] = measure q[75];
c[76] = measure q[76];
c[77] = measure q[77];
c[78] = measure q[78];
c[79] = measure q[79];