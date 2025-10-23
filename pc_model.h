// Auto-generated Decision Tree for 8051 (Keil)
// Task: classify pre/post Ziegler–Nichols (0=pre, 1=post)
#ifndef PC_MODEL_H
#define PC_MODEL_H

#define NODES (7)
#define FEATURES (3)

static const char feat[NODES] = {
  1, 2, -2, -2, 1, -2, -2
};

static const unsigned char thresh[NODES] = {
  72, 80, 0, 0, 148, 0, 0
};

static const long cleft[NODES] = {
  1, 2, -1, -1, 5, -1, -1
};

static const long cright[NODES] = {
  4, 3, -1, -1, 6, -1, -1
};

static const unsigned char value[NODES] = {
  0, 0, 1, 0, 1, 1, 0
};

static const unsigned int MODEL_CRC16 = 0xFDC7;

#endif // PC_MODEL_H