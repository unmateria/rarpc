__device__ __forceinline__ void switch_buf_carry_be_full(uint32_t w0[4], uint32_t w1[4], uint32_t w2[4], uint32_t w3[4], uint32_t c0[4], uint32_t c1[4], uint32_t c2[4], uint32_t c3[4], uint32_t offset)
{
  const int sw = (int)(offset >> 2);

  switch (sw)
  {
    case  0:
      c0[0] = hc_bytealign_be_S (w3[3],     0, offset);
      w3[3] = hc_bytealign_be_S (w3[2], w3[3], offset);
      w3[2] = hc_bytealign_be_S (w3[1], w3[2], offset);
      w3[1] = hc_bytealign_be_S (w3[0], w3[1], offset);
      w3[0] = hc_bytealign_be_S (w2[3], w3[0], offset);
      w2[3] = hc_bytealign_be_S (w2[2], w2[3], offset);
      w2[2] = hc_bytealign_be_S (w2[1], w2[2], offset);
      w2[1] = hc_bytealign_be_S (w2[0], w2[1], offset);
      w2[0] = hc_bytealign_be_S (w1[3], w2[0], offset);
      w1[3] = hc_bytealign_be_S (w1[2], w1[3], offset);
      w1[2] = hc_bytealign_be_S (w1[1], w1[2], offset);
      w1[1] = hc_bytealign_be_S (w1[0], w1[1], offset);
      w1[0] = hc_bytealign_be_S (w0[3], w1[0], offset);
      w0[3] = hc_bytealign_be_S (w0[2], w0[3], offset);
      w0[2] = hc_bytealign_be_S (w0[1], w0[2], offset);
      w0[1] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w0[0] = hc_bytealign_be_S (    0, w0[0], offset);

      break;

    case  1:
      c0[1] = hc_bytealign_be_S (w3[3],     0, offset);
      c0[0] = hc_bytealign_be_S (w3[2], w3[3], offset);
      w3[3] = hc_bytealign_be_S (w3[1], w3[2], offset);
      w3[2] = hc_bytealign_be_S (w3[0], w3[1], offset);
      w3[1] = hc_bytealign_be_S (w2[3], w3[0], offset);
      w3[0] = hc_bytealign_be_S (w2[2], w2[3], offset);
      w2[3] = hc_bytealign_be_S (w2[1], w2[2], offset);
      w2[2] = hc_bytealign_be_S (w2[0], w2[1], offset);
      w2[1] = hc_bytealign_be_S (w1[3], w2[0], offset);
      w2[0] = hc_bytealign_be_S (w1[2], w1[3], offset);
      w1[3] = hc_bytealign_be_S (w1[1], w1[2], offset);
      w1[2] = hc_bytealign_be_S (w1[0], w1[1], offset);
      w1[1] = hc_bytealign_be_S (w0[3], w1[0], offset);
      w1[0] = hc_bytealign_be_S (w0[2], w0[3], offset);
      w0[3] = hc_bytealign_be_S (w0[1], w0[2], offset);
      w0[2] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w0[1] = hc_bytealign_be_S (    0, w0[0], offset);
      w0[0] = 0;

      break;

    case  2:
      c0[2] = hc_bytealign_be_S (w3[3],     0, offset);
      c0[1] = hc_bytealign_be_S (w3[2], w3[3], offset);
      c0[0] = hc_bytealign_be_S (w3[1], w3[2], offset);
      w3[3] = hc_bytealign_be_S (w3[0], w3[1], offset);
      w3[2] = hc_bytealign_be_S (w2[3], w3[0], offset);
      w3[1] = hc_bytealign_be_S (w2[2], w2[3], offset);
      w3[0] = hc_bytealign_be_S (w2[1], w2[2], offset);
      w2[3] = hc_bytealign_be_S (w2[0], w2[1], offset);
      w2[2] = hc_bytealign_be_S (w1[3], w2[0], offset);
      w2[1] = hc_bytealign_be_S (w1[2], w1[3], offset);
      w2[0] = hc_bytealign_be_S (w1[1], w1[2], offset);
      w1[3] = hc_bytealign_be_S (w1[0], w1[1], offset);
      w1[2] = hc_bytealign_be_S (w0[3], w1[0], offset);
      w1[1] = hc_bytealign_be_S (w0[2], w0[3], offset);
      w1[0] = hc_bytealign_be_S (w0[1], w0[2], offset);
      w0[3] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w0[2] = hc_bytealign_be_S (    0, w0[0], offset);
      w0[1] = 0;
      w0[0] = 0;

      break;

    case  3:
      c0[3] = hc_bytealign_be_S (w3[3],     0, offset);
      c0[2] = hc_bytealign_be_S (w3[2], w3[3], offset);
      c0[1] = hc_bytealign_be_S (w3[1], w3[2], offset);
      c0[0] = hc_bytealign_be_S (w3[0], w3[1], offset);
      w3[3] = hc_bytealign_be_S (w2[3], w3[0], offset);
      w3[2] = hc_bytealign_be_S (w2[2], w2[3], offset);
      w3[1] = hc_bytealign_be_S (w2[1], w2[2], offset);
      w3[0] = hc_bytealign_be_S (w2[0], w2[1], offset);
      w2[3] = hc_bytealign_be_S (w1[3], w2[0], offset);
      w2[2] = hc_bytealign_be_S (w1[2], w1[3], offset);
      w2[1] = hc_bytealign_be_S (w1[1], w1[2], offset);
      w2[0] = hc_bytealign_be_S (w1[0], w1[1], offset);
      w1[3] = hc_bytealign_be_S (w0[3], w1[0], offset);
      w1[2] = hc_bytealign_be_S (w0[2], w0[3], offset);
      w1[1] = hc_bytealign_be_S (w0[1], w0[2], offset);
      w1[0] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w0[3] = hc_bytealign_be_S (    0, w0[0], offset);
      w0[2] = 0;
      w0[1] = 0;
      w0[0] = 0;

      break;

    case  4:
      c1[0] = hc_bytealign_be_S (w3[3],     0, offset);
      c0[3] = hc_bytealign_be_S (w3[2], w3[3], offset);
      c0[2] = hc_bytealign_be_S (w3[1], w3[2], offset);
      c0[1] = hc_bytealign_be_S (w3[0], w3[1], offset);
      c0[0] = hc_bytealign_be_S (w2[3], w3[0], offset);
      w3[3] = hc_bytealign_be_S (w2[2], w2[3], offset);
      w3[2] = hc_bytealign_be_S (w2[1], w2[2], offset);
      w3[1] = hc_bytealign_be_S (w2[0], w2[1], offset);
      w3[0] = hc_bytealign_be_S (w1[3], w2[0], offset);
      w2[3] = hc_bytealign_be_S (w1[2], w1[3], offset);
      w2[2] = hc_bytealign_be_S (w1[1], w1[2], offset);
      w2[1] = hc_bytealign_be_S (w1[0], w1[1], offset);
      w2[0] = hc_bytealign_be_S (w0[3], w1[0], offset);
      w1[3] = hc_bytealign_be_S (w0[2], w0[3], offset);
      w1[2] = hc_bytealign_be_S (w0[1], w0[2], offset);
      w1[1] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w1[0] = hc_bytealign_be_S (    0, w0[0], offset);
      w0[3] = 0;
      w0[2] = 0;
      w0[1] = 0;
      w0[0] = 0;

      break;

    case  5:
      c1[1] = hc_bytealign_be_S (w3[3],     0, offset);
      c1[0] = hc_bytealign_be_S (w3[2], w3[3], offset);
      c0[3] = hc_bytealign_be_S (w3[1], w3[2], offset);
      c0[2] = hc_bytealign_be_S (w3[0], w3[1], offset);
      c0[1] = hc_bytealign_be_S (w2[3], w3[0], offset);
      c0[0] = hc_bytealign_be_S (w2[2], w2[3], offset);
      w3[3] = hc_bytealign_be_S (w2[1], w2[2], offset);
      w3[2] = hc_bytealign_be_S (w2[0], w2[1], offset);
      w3[1] = hc_bytealign_be_S (w1[3], w2[0], offset);
      w3[0] = hc_bytealign_be_S (w1[2], w1[3], offset);
      w2[3] = hc_bytealign_be_S (w1[1], w1[2], offset);
      w2[2] = hc_bytealign_be_S (w1[0], w1[1], offset);
      w2[1] = hc_bytealign_be_S (w0[3], w1[0], offset);
      w2[0] = hc_bytealign_be_S (w0[2], w0[3], offset);
      w1[3] = hc_bytealign_be_S (w0[1], w0[2], offset);
      w1[2] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w1[1] = hc_bytealign_be_S (    0, w0[0], offset);
      w1[0] = 0;
      w0[3] = 0;
      w0[2] = 0;
      w0[1] = 0;
      w0[0] = 0;

      break;

    case  6:
      c1[2] = hc_bytealign_be_S (w3[3],     0, offset);
      c1[1] = hc_bytealign_be_S (w3[2], w3[3], offset);
      c1[0] = hc_bytealign_be_S (w3[1], w3[2], offset);
      c0[3] = hc_bytealign_be_S (w3[0], w3[1], offset);
      c0[2] = hc_bytealign_be_S (w2[3], w3[0], offset);
      c0[1] = hc_bytealign_be_S (w2[2], w2[3], offset);
      c0[0] = hc_bytealign_be_S (w2[1], w2[2], offset);
      w3[3] = hc_bytealign_be_S (w2[0], w2[1], offset);
      w3[2] = hc_bytealign_be_S (w1[3], w2[0], offset);
      w3[1] = hc_bytealign_be_S (w1[2], w1[3], offset);
      w3[0] = hc_bytealign_be_S (w1[1], w1[2], offset);
      w2[3] = hc_bytealign_be_S (w1[0], w1[1], offset);
      w2[2] = hc_bytealign_be_S (w0[3], w1[0], offset);
      w2[1] = hc_bytealign_be_S (w0[2], w0[3], offset);
      w2[0] = hc_bytealign_be_S (w0[1], w0[2], offset);
      w1[3] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w1[2] = hc_bytealign_be_S (    0, w0[0], offset);
      w1[1] = 0;
      w1[0] = 0;
      w0[3] = 0;
      w0[2] = 0;
      w0[1] = 0;
      w0[0] = 0;

      break;

    case  7:
      c1[3] = hc_bytealign_be_S (w3[3],     0, offset);
      c1[2] = hc_bytealign_be_S (w3[2], w3[3], offset);
      c1[1] = hc_bytealign_be_S (w3[1], w3[2], offset);
      c1[0] = hc_bytealign_be_S (w3[0], w3[1], offset);
      c0[3] = hc_bytealign_be_S (w2[3], w3[0], offset);
      c0[2] = hc_bytealign_be_S (w2[2], w2[3], offset);
      c0[1] = hc_bytealign_be_S (w2[1], w2[2], offset);
      c0[0] = hc_bytealign_be_S (w2[0], w2[1], offset);
      w3[3] = hc_bytealign_be_S (w1[3], w2[0], offset);
      w3[2] = hc_bytealign_be_S (w1[2], w1[3], offset);
      w3[1] = hc_bytealign_be_S (w1[1], w1[2], offset);
      w3[0] = hc_bytealign_be_S (w1[0], w1[1], offset);
      w2[3] = hc_bytealign_be_S (w0[3], w1[0], offset);
      w2[2] = hc_bytealign_be_S (w0[2], w0[3], offset);
      w2[1] = hc_bytealign_be_S (w0[1], w0[2], offset);
      w2[0] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w1[3] = hc_bytealign_be_S (    0, w0[0], offset);
      w1[2] = 0;
      w1[1] = 0;
      w1[0] = 0;
      w0[3] = 0;
      w0[2] = 0;
      w0[1] = 0;
      w0[0] = 0;

      break;

    case  8:
      c2[0] = hc_bytealign_be_S (w3[3],     0, offset);
      c1[3] = hc_bytealign_be_S (w3[2], w3[3], offset);
      c1[2] = hc_bytealign_be_S (w3[1], w3[2], offset);
      c1[1] = hc_bytealign_be_S (w3[0], w3[1], offset);
      c1[0] = hc_bytealign_be_S (w2[3], w3[0], offset);
      c0[3] = hc_bytealign_be_S (w2[2], w2[3], offset);
      c0[2] = hc_bytealign_be_S (w2[1], w2[2], offset);
      c0[1] = hc_bytealign_be_S (w2[0], w2[1], offset);
      c0[0] = hc_bytealign_be_S (w1[3], w2[0], offset);
      w3[3] = hc_bytealign_be_S (w1[2], w1[3], offset);
      w3[2] = hc_bytealign_be_S (w1[1], w1[2], offset);
      w3[1] = hc_bytealign_be_S (w1[0], w1[1], offset);
      w3[0] = hc_bytealign_be_S (w0[3], w1[0], offset);
      w2[3] = hc_bytealign_be_S (w0[2], w0[3], offset);
      w2[2] = hc_bytealign_be_S (w0[1], w0[2], offset);
      w2[1] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w2[0] = hc_bytealign_be_S (    0, w0[0], offset);
      w1[3] = 0;
      w1[2] = 0;
      w1[1] = 0;
      w1[0] = 0;
      w0[3] = 0;
      w0[2] = 0;
      w0[1] = 0;
      w0[0] = 0;

      break;

    case  9:
      c2[1] = hc_bytealign_be_S (w3[3],     0, offset);
      c2[0] = hc_bytealign_be_S (w3[2], w3[3], offset);
      c1[3] = hc_bytealign_be_S (w3[1], w3[2], offset);
      c1[2] = hc_bytealign_be_S (w3[0], w3[1], offset);
      c1[1] = hc_bytealign_be_S (w2[3], w3[0], offset);
      c1[0] = hc_bytealign_be_S (w2[2], w2[3], offset);
      c0[3] = hc_bytealign_be_S (w2[1], w2[2], offset);
      c0[2] = hc_bytealign_be_S (w2[0], w2[1], offset);
      c0[1] = hc_bytealign_be_S (w1[3], w2[0], offset);
      c0[0] = hc_bytealign_be_S (w1[2], w1[3], offset);
      w3[3] = hc_bytealign_be_S (w1[1], w1[2], offset);
      w3[2] = hc_bytealign_be_S (w1[0], w1[1], offset);
      w3[1] = hc_bytealign_be_S (w0[3], w1[0], offset);
      w3[0] = hc_bytealign_be_S (w0[2], w0[3], offset);
      w2[3] = hc_bytealign_be_S (w0[1], w0[2], offset);
      w2[2] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w2[1] = hc_bytealign_be_S (    0, w0[0], offset);
      w2[0] = 0;
      w1[3] = 0;
      w1[2] = 0;
      w1[1] = 0;
      w1[0] = 0;
      w0[3] = 0;
      w0[2] = 0;
      w0[1] = 0;
      w0[0] = 0;

      break;

    case 10:
      c2[2] = hc_bytealign_be_S (w3[3],     0, offset);
      c2[1] = hc_bytealign_be_S (w3[2], w3[3], offset);
      c2[0] = hc_bytealign_be_S (w3[1], w3[2], offset);
      c1[3] = hc_bytealign_be_S (w3[0], w3[1], offset);
      c1[2] = hc_bytealign_be_S (w2[3], w3[0], offset);
      c1[1] = hc_bytealign_be_S (w2[2], w2[3], offset);
      c1[0] = hc_bytealign_be_S (w2[1], w2[2], offset);
      c0[3] = hc_bytealign_be_S (w2[0], w2[1], offset);
      c0[2] = hc_bytealign_be_S (w1[3], w2[0], offset);
      c0[1] = hc_bytealign_be_S (w1[2], w1[3], offset);
      c0[0] = hc_bytealign_be_S (w1[1], w1[2], offset);
      w3[3] = hc_bytealign_be_S (w1[0], w1[1], offset);
      w3[2] = hc_bytealign_be_S (w0[3], w1[0], offset);
      w3[1] = hc_bytealign_be_S (w0[2], w0[3], offset);
      w3[0] = hc_bytealign_be_S (w0[1], w0[2], offset);
      w2[3] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w2[2] = hc_bytealign_be_S (    0, w0[0], offset);
      w2[1] = 0;
      w2[0] = 0;
      w1[3] = 0;
      w1[2] = 0;
      w1[1] = 0;
      w1[0] = 0;
      w0[3] = 0;
      w0[2] = 0;
      w0[1] = 0;
      w0[0] = 0;

      break;

    case 11:
      c2[3] = hc_bytealign_be_S (w3[3],     0, offset);
      c2[2] = hc_bytealign_be_S (w3[2], w3[3], offset);
      c2[1] = hc_bytealign_be_S (w3[1], w3[2], offset);
      c2[0] = hc_bytealign_be_S (w3[0], w3[1], offset);
      c1[3] = hc_bytealign_be_S (w2[3], w3[0], offset);
      c1[2] = hc_bytealign_be_S (w2[2], w2[3], offset);
      c1[1] = hc_bytealign_be_S (w2[1], w2[2], offset);
      c1[0] = hc_bytealign_be_S (w2[0], w2[1], offset);
      c0[3] = hc_bytealign_be_S (w1[3], w2[0], offset);
      c0[2] = hc_bytealign_be_S (w1[2], w1[3], offset);
      c0[1] = hc_bytealign_be_S (w1[1], w1[2], offset);
      c0[0] = hc_bytealign_be_S (w1[0], w1[1], offset);
      w3[3] = hc_bytealign_be_S (w0[3], w1[0], offset);
      w3[2] = hc_bytealign_be_S (w0[2], w0[3], offset);
      w3[1] = hc_bytealign_be_S (w0[1], w0[2], offset);
      w3[0] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w2[3] = hc_bytealign_be_S (    0, w0[0], offset);
      w2[2] = 0;
      w2[1] = 0;
      w2[0] = 0;
      w1[3] = 0;
      w1[2] = 0;
      w1[1] = 0;
      w1[0] = 0;
      w0[3] = 0;
      w0[2] = 0;
      w0[1] = 0;
      w0[0] = 0;

      break;

    case 12:
      c3[0] = hc_bytealign_be_S (w3[3],     0, offset);
      c2[3] = hc_bytealign_be_S (w3[2], w3[3], offset);
      c2[2] = hc_bytealign_be_S (w3[1], w3[2], offset);
      c2[1] = hc_bytealign_be_S (w3[0], w3[1], offset);
      c2[0] = hc_bytealign_be_S (w2[3], w3[0], offset);
      c1[3] = hc_bytealign_be_S (w2[2], w2[3], offset);
      c1[2] = hc_bytealign_be_S (w2[1], w2[2], offset);
      c1[1] = hc_bytealign_be_S (w2[0], w2[1], offset);
      c1[0] = hc_bytealign_be_S (w1[3], w2[0], offset);
      c0[3] = hc_bytealign_be_S (w1[2], w1[3], offset);
      c0[2] = hc_bytealign_be_S (w1[1], w1[2], offset);
      c0[1] = hc_bytealign_be_S (w1[0], w1[1], offset);
      c0[0] = hc_bytealign_be_S (w0[3], w1[0], offset);
      w3[3] = hc_bytealign_be_S (w0[2], w0[3], offset);
      w3[2] = hc_bytealign_be_S (w0[1], w0[2], offset);
      w3[1] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w3[0] = hc_bytealign_be_S (    0, w0[0], offset);
      w2[3] = 0;
      w2[2] = 0;
      w2[1] = 0;
      w2[0] = 0;
      w1[3] = 0;
      w1[2] = 0;
      w1[1] = 0;
      w1[0] = 0;
      w0[3] = 0;
      w0[2] = 0;
      w0[1] = 0;
      w0[0] = 0;

      break;

    case 13:
      c3[1] = hc_bytealign_be_S (w3[3],     0, offset);
      c3[0] = hc_bytealign_be_S (w3[2], w3[3], offset);
      c2[3] = hc_bytealign_be_S (w3[1], w3[2], offset);
      c2[2] = hc_bytealign_be_S (w3[0], w3[1], offset);
      c2[1] = hc_bytealign_be_S (w2[3], w3[0], offset);
      c2[0] = hc_bytealign_be_S (w2[2], w2[3], offset);
      c1[3] = hc_bytealign_be_S (w2[1], w2[2], offset);
      c1[2] = hc_bytealign_be_S (w2[0], w2[1], offset);
      c1[1] = hc_bytealign_be_S (w1[3], w2[0], offset);
      c1[0] = hc_bytealign_be_S (w1[2], w1[3], offset);
      c0[3] = hc_bytealign_be_S (w1[1], w1[2], offset);
      c0[2] = hc_bytealign_be_S (w1[0], w1[1], offset);
      c0[1] = hc_bytealign_be_S (w0[3], w1[0], offset);
      c0[0] = hc_bytealign_be_S (w0[2], w0[3], offset);
      w3[3] = hc_bytealign_be_S (w0[1], w0[2], offset);
      w3[2] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w3[1] = hc_bytealign_be_S (    0, w0[0], offset);
      w3[0] = 0;
      w2[3] = 0;
      w2[2] = 0;
      w2[1] = 0;
      w2[0] = 0;
      w1[3] = 0;
      w1[2] = 0;
      w1[1] = 0;
      w1[0] = 0;
      w0[3] = 0;
      w0[2] = 0;
      w0[1] = 0;
      w0[0] = 0;

      break;

    case 14:
      c3[2] = hc_bytealign_be_S (w3[3],     0, offset);
      c3[1] = hc_bytealign_be_S (w3[2], w3[3], offset);
      c3[0] = hc_bytealign_be_S (w3[1], w3[2], offset);
      c2[3] = hc_bytealign_be_S (w3[0], w3[1], offset);
      c2[2] = hc_bytealign_be_S (w2[3], w3[0], offset);
      c2[1] = hc_bytealign_be_S (w2[2], w2[3], offset);
      c2[0] = hc_bytealign_be_S (w2[1], w2[2], offset);
      c1[3] = hc_bytealign_be_S (w2[0], w2[1], offset);
      c1[2] = hc_bytealign_be_S (w1[3], w2[0], offset);
      c1[1] = hc_bytealign_be_S (w1[2], w1[3], offset);
      c1[0] = hc_bytealign_be_S (w1[1], w1[2], offset);
      c0[3] = hc_bytealign_be_S (w1[0], w1[1], offset);
      c0[2] = hc_bytealign_be_S (w0[3], w1[0], offset);
      c0[1] = hc_bytealign_be_S (w0[2], w0[3], offset);
      c0[0] = hc_bytealign_be_S (w0[1], w0[2], offset);
      w3[3] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w3[2] = hc_bytealign_be_S (    0, w0[0], offset);
      w3[1] = 0;
      w3[0] = 0;
      w2[3] = 0;
      w2[2] = 0;
      w2[1] = 0;
      w2[0] = 0;
      w1[3] = 0;
      w1[2] = 0;
      w1[1] = 0;
      w1[0] = 0;
      w0[3] = 0;
      w0[2] = 0;
      w0[1] = 0;
      w0[0] = 0;

      break;

    case 15:
      c3[3] = hc_bytealign_be_S (w3[3],     0, offset);
      c3[2] = hc_bytealign_be_S (w3[2], w3[3], offset);
      c3[1] = hc_bytealign_be_S (w3[1], w3[2], offset);
      c3[0] = hc_bytealign_be_S (w3[0], w3[1], offset);
      c2[3] = hc_bytealign_be_S (w2[3], w3[0], offset);
      c2[2] = hc_bytealign_be_S (w2[2], w2[3], offset);
      c2[1] = hc_bytealign_be_S (w2[1], w2[2], offset);
      c2[0] = hc_bytealign_be_S (w2[0], w2[1], offset);
      c1[3] = hc_bytealign_be_S (w1[3], w2[0], offset);
      c1[2] = hc_bytealign_be_S (w1[2], w1[3], offset);
      c1[1] = hc_bytealign_be_S (w1[1], w1[2], offset);
      c1[0] = hc_bytealign_be_S (w1[0], w1[1], offset);
      c0[3] = hc_bytealign_be_S (w0[3], w1[0], offset);
      c0[2] = hc_bytealign_be_S (w0[2], w0[3], offset);
      c0[1] = hc_bytealign_be_S (w0[1], w0[2], offset);
      c0[0] = hc_bytealign_be_S (w0[0], w0[1], offset);
      w3[3] = hc_bytealign_be_S (    0, w0[0], offset);
      w3[2] = 0;
      w3[1] = 0;
      w3[0] = 0;
      w2[3] = 0;
      w2[2] = 0;
      w2[1] = 0;
      w2[0] = 0;
      w1[3] = 0;
      w1[2] = 0;
      w1[1] = 0;
      w1[0] = 0;
      w0[3] = 0;
      w0[2] = 0;
      w0[1] = 0;
      w0[0] = 0;

      break;
  }
}
