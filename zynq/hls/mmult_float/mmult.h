
#include <ap_axi_sdata.h>
#include <assert.h>
#include <hls_stream.h>

typedef unsigned long long axi_T;
typedef float T;

// Matrix dimensions specifications
#define BATCH 8 // TODO: you will tweak this later
#define FEAT 256
#define CLASSES 10

// Input/Output Stream Size
#define IS_SIZE (BATCH * FEAT / WIDTH_RATIO + (FEAT + 1) * CLASSES / WIDTH_RATIO)
#define OS_SIZE (BATCH * CLASSES / WIDTH_RATIO)

// AXI settings
#define AXI_DATA (sizeof(axi_T) * 8)
#define AXI_U 4
#define AXI_TI 5
#define AXI_TD 5

// Data type ratio between data type and axi width
#define WIDTH_RATIO (sizeof(axi_T) / sizeof(T))

typedef ap_axiu<AXI_DATA, AXI_U, AXI_TI, AXI_TD> AXI_VAL;

// Matrix Multiply prototype
void mmult_hw(hls::stream<AXI_VAL>& in_stream, hls::stream<AXI_VAL>& out_stream);

// AXI stream push and pop
axi_T pop_stream(hls::stream<AXI_VAL>& in_stream);
AXI_VAL push_stream(axi_T const& v, bool last);
