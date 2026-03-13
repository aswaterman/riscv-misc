#include <stdio.h>

unsigned punning (
  const unsigned TE,  // tile array dimension TExTE
  unsigned tile,      // addressed tile
  unsigned row,       // tile row index
  unsigned col,       // tile col index
  unsigned tew        // tile element width
) {
  unsigned ptile, minor_offset, major_offset;

  switch (tew) {
    case 8: {
      ptile = tile;
      minor_offset = (row % 4) * 4 + (col % 4);
      major_offset = (row / 4) * (TE / 4) + (col / 4);
      break;
    }
    case 16: {
      ptile = tile + ((row & 2) >> 1);
      minor_offset = (row % 2) * 4 + (col % 2) * 2 + ((col / 2) % 2) * 8;
      major_offset = (row / 4) * (TE / 4) + (col / 4);
      break;
    }
    case 32: {
      ptile = tile + 2 * (row / (TE / 2)) + ((col & 2) >> 1);
      minor_offset = (row % 2) * 8 + (col % 2) * 4;
      major_offset = ((row / 2) % (TE / 4)) * (TE / 4) + (col / 4);
      break;
    }
    case 64: {
      ptile = tile + (row / (TE / 4));
      minor_offset = (col % 2) * 8;
      major_offset = (row % (TE / 4)) * (TE / 4) + (col / 2);
      break;
    }
    default: return 0;
  }

  unsigned offset = (ptile * TE * TE) + (major_offset * 16) + minor_offset;

  return offset;
}

int main() {
  unsigned TE = 16;
  printf ("TE: %u\n", TE);
  for (unsigned tew=8; tew<128; tew<<=1) {
    printf ("tew: %u\n", tew);
    for (unsigned t=0; t<4; t++) {
      unsigned tile = t;
      unsigned nTE = TE;
      switch (tew) {
       default: { // case 8
         break;
       }
       case 16: {
         if (t > 1) continue;
         tile = t << 1;
         break;
       }
       case 32: {
         if (t > 0) continue;
         break;
       }
       case 64: {
         if (t > 1) continue;
         tile = t << 1;
         nTE = TE/2;
         break;
       }
      }
      printf ("  tile: %u\n", tile);
      for (unsigned row=0; row<nTE; row++) {
        printf ("    [%2u,0]", row);
        for (unsigned col=0; col<nTE; col++) {
          unsigned off = punning(TE,tile,row,col,tew);
          printf (" 0x%03x", off);
        }
        printf ("\n");
      }
    }
  }
  return 0;
}
