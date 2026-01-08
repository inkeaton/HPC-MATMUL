	.text
	.file	"matmul_seq_ikj.c"
	.file	1 "/home/Zsf/Desktop/_FINAL_PROJECT" "src/matmul_seq_ikj.c"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function main
.LCPI0_0:
	.quad	0x4000000000000000              #  2
.LCPI0_1:
	.quad	0x4008000000000000              #  3
.LCPI0_2:
	.quad	0x3e112e0be826d695              #  1.0000000000000001E-9
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # 
.Lfunc_begin0:
	.loc	1 39 0                          # src/matmul_seq_ikj.c:39:0
	.cfi_startproc
# %bb.0:
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$280, %rsp                      # imm = 0x118
	.cfi_def_cfa_offset 336
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movabsq	$4503599906136046, %rsi         # imm = 0x100000109D9FEE
.Ltmp0:
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	movl	$3, %edi
.Ltmp1:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	callq	__intel_new_feature_proc_init@PLT
.Ltmp2:
	.loc	1 47 32 prologue_end            # src/matmul_seq_ikj.c:47:32
	movl	$64, %edi
	movl	$200000000, %esi                # imm = 0xBEBC200
	callq	aligned_alloc
.Ltmp3:
	movq	%rax, %r15
.Ltmp4:
	#DEBUG_VALUE: main:a <- $r15
	.loc	1 48 32                         # src/matmul_seq_ikj.c:48:32
	movl	$64, %edi
	movl	$200000000, %esi                # imm = 0xBEBC200
	callq	aligned_alloc
.Ltmp5:
	movq	%rax, %r14
.Ltmp6:
	#DEBUG_VALUE: main:b <- $r14
	.loc	1 49 32                         # src/matmul_seq_ikj.c:49:32
	movl	$64, %edi
	movl	$200000000, %esi                # imm = 0xBEBC200
	callq	aligned_alloc
.Ltmp7:
	#DEBUG_VALUE: main:c <- $rax
	.loc	1 51 10                         # src/matmul_seq_ikj.c:51:10
	testq	%r15, %r15
	.loc	1 51 12 is_stmt 0               # src/matmul_seq_ikj.c:51:12
	je	.LBB0_21
.Ltmp8:
# %bb.1:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $rax
	testq	%r14, %r14
	je	.LBB0_21
.Ltmp9:
# %bb.2:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $rax
	.loc	1 49 32 is_stmt 1               # src/matmul_seq_ikj.c:49:32
	movq	%rax, %r12
.Ltmp10:
	.loc	1 51 12                         # src/matmul_seq_ikj.c:51:12
	testq	%rax, %rax
	je	.LBB0_21
.Ltmp11:
# %bb.3:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r12
	.loc	1 63 21                         # src/matmul_seq_ikj.c:63:21
	movl	$200000000, %edx                # imm = 0xBEBC200
	movq	%r12, %rdi
	xorl	%esi, %esi
	callq	_intel_fast_memset@PLT
.Ltmp12:
	.loc	1 0 21 is_stmt 0                # src/matmul_seq_ikj.c:0:21
	movq	$-40, %rax
	.loc	1 61 21 is_stmt 1 discriminator 2 # src/matmul_seq_ikj.c:61:21
	vbroadcastsd	.LCPI0_0(%rip), %ymm0   # ymm0 = [2.0E+0,2.0E+0,2.0E+0,2.0E+0]
	.loc	1 62 21 discriminator 2         # src/matmul_seq_ikj.c:62:21
	vbroadcastsd	.LCPI0_1(%rip), %ymm1   # ymm1 = [3.0E+0,3.0E+0,3.0E+0,3.0E+0]
.Ltmp13:
	.p2align	4, 0x90
.LBB0_4:                                # =>This Inner Loop Header: Depth=1
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r12
	.loc	1 61 21                         # src/matmul_seq_ikj.c:61:21
	vmovntpd	%ymm0, 320(%r15,%rax,8)
	.loc	1 62 21                         # src/matmul_seq_ikj.c:62:21
	vmovntpd	%ymm1, 320(%r14,%rax,8)
	.loc	1 61 21                         # src/matmul_seq_ikj.c:61:21
	vmovntpd	%ymm0, 352(%r15,%rax,8)
	.loc	1 62 21                         # src/matmul_seq_ikj.c:62:21
	vmovntpd	%ymm1, 352(%r14,%rax,8)
	.loc	1 61 21                         # src/matmul_seq_ikj.c:61:21
	vmovntpd	%ymm0, 384(%r15,%rax,8)
	.loc	1 62 21                         # src/matmul_seq_ikj.c:62:21
	vmovntpd	%ymm1, 384(%r14,%rax,8)
	.loc	1 61 21                         # src/matmul_seq_ikj.c:61:21
	vmovntpd	%ymm0, 416(%r15,%rax,8)
	.loc	1 62 21                         # src/matmul_seq_ikj.c:62:21
	vmovntpd	%ymm1, 416(%r14,%rax,8)
	.loc	1 61 21                         # src/matmul_seq_ikj.c:61:21
	vmovntpd	%ymm0, 448(%r15,%rax,8)
	.loc	1 62 21                         # src/matmul_seq_ikj.c:62:21
	vmovntpd	%ymm1, 448(%r14,%rax,8)
	.loc	1 61 21                         # src/matmul_seq_ikj.c:61:21
	vmovntpd	%ymm0, 480(%r15,%rax,8)
	.loc	1 62 21                         # src/matmul_seq_ikj.c:62:21
	vmovntpd	%ymm1, 480(%r14,%rax,8)
	.loc	1 61 21                         # src/matmul_seq_ikj.c:61:21
	vmovntpd	%ymm0, 512(%r15,%rax,8)
	.loc	1 62 21                         # src/matmul_seq_ikj.c:62:21
	vmovntpd	%ymm1, 512(%r14,%rax,8)
	.loc	1 61 21                         # src/matmul_seq_ikj.c:61:21
	vmovntpd	%ymm0, 544(%r15,%rax,8)
	.loc	1 62 21                         # src/matmul_seq_ikj.c:62:21
	vmovntpd	%ymm1, 544(%r14,%rax,8)
	.loc	1 61 21                         # src/matmul_seq_ikj.c:61:21
	vmovntpd	%ymm0, 576(%r15,%rax,8)
	.loc	1 62 21                         # src/matmul_seq_ikj.c:62:21
	vmovntpd	%ymm1, 576(%r14,%rax,8)
	.loc	1 61 21                         # src/matmul_seq_ikj.c:61:21
	vmovntpd	%ymm0, 608(%r15,%rax,8)
	.loc	1 62 21                         # src/matmul_seq_ikj.c:62:21
	vmovntpd	%ymm1, 608(%r14,%rax,8)
.Ltmp14:
	.loc	1 59 27                         # src/matmul_seq_ikj.c:59:27
	addq	$40, %rax
	cmpq	$24999960, %rax                 # imm = 0x17D7818
.Ltmp15:
	.loc	1 59 9 is_stmt 0                # src/matmul_seq_ikj.c:59:9
	jb	.LBB0_4
.Ltmp16:
# %bb.5:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r12
	.loc	1 51 12 is_stmt 1               # src/matmul_seq_ikj.c:51:12
	sfence
	leaq	96(%rsp), %rsi
.Ltmp17:
	.loc	1 69 9                          # src/matmul_seq_ikj.c:69:9
	movl	$1, %edi
	vzeroupper
	callq	clock_gettime
.Ltmp18:
	#DEBUG_VALUE: i <- 0
	.loc	1 0 9 is_stmt 0                 # src/matmul_seq_ikj.c:0:9
	movq	%r12, 8(%rsp)                   # 8-byte Spill
.Ltmp19:
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	leaq	120000(%r12), %rax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movq	%r14, 32(%rsp)                  # 8-byte Spill
.Ltmp20:
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	addq	$120000, %r14                   # imm = 0x1D4C0
	movq	%r14, 40(%rsp)                  # 8-byte Spill
	xorl	%edi, %edi
	jmp	.LBB0_6
.Ltmp21:
	.p2align	4, 0x90
.LBB0_18:                               #   in Loop: Header=BB0_6 Depth=1
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	#DEBUG_VALUE: i <- 0
	.loc	1 73 5 is_stmt 1                # src/matmul_seq_ikj.c:73:5
	addq	$2560000, 16(%rsp)              # 8-byte Folded Spill
                                        # imm = 0x271000
	movq	48(%rsp), %rdi                  # 8-byte Reload
.Ltmp22:
	.loc	1 73 23 is_stmt 0               # src/matmul_seq_ikj.c:73:23
	cmpq	$78, %rdi
	leaq	1(%rdi), %rdi
.Ltmp23:
	.loc	1 73 5                          # src/matmul_seq_ikj.c:73:5
	je	.LBB0_19
.Ltmp24:
.LBB0_6:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_7 Depth 2
                                        #       Child Loop BB0_8 Depth 3
                                        #         Child Loop BB0_10 Depth 4
                                        #           Child Loop BB0_12 Depth 5
                                        #             Child Loop BB0_13 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	.loc	1 0 5                           # src/matmul_seq_ikj.c:0:5
	movq	%rdi, 48(%rsp)                  # 8-byte Spill
.Ltmp25:
	.loc	1 51 12 is_stmt 1               # src/matmul_seq_ikj.c:51:12
	shlq	$6, %rdi
	movl	$4999, %r9d                     # imm = 0x1387
	subq	%rdi, %r9
	cmpq	$64, %r9
	movl	$63, %eax
	cmovgeq	%rax, %r9
	incq	%r9
	movq	%r9, 72(%rsp)                   # 8-byte Spill
	shrq	$2, %r9
	decq	%r9
	movq	40(%rsp), %rsi                  # 8-byte Reload
	xorl	%r12d, %r12d
	jmp	.LBB0_7
.Ltmp26:
	.p2align	4, 0x90
.LBB0_17:                               #   in Loop: Header=BB0_7 Depth=2
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	#DEBUG_VALUE: i <- 0
	.loc	1 0 12 is_stmt 0                # src/matmul_seq_ikj.c:0:12
	movq	64(%rsp), %rsi                  # 8-byte Reload
.Ltmp27:
	.loc	1 73 5 is_stmt 1                # src/matmul_seq_ikj.c:73:5
	addq	$2560000, %rsi                  # imm = 0x271000
	movq	56(%rsp), %r12                  # 8-byte Reload
.Ltmp28:
	.loc	1 73 23 is_stmt 0               # src/matmul_seq_ikj.c:73:23
	cmpq	$78, %r12
	leaq	1(%r12), %r12
.Ltmp29:
	.loc	1 73 5                          # src/matmul_seq_ikj.c:73:5
	je	.LBB0_18
.Ltmp30:
.LBB0_7:                                #   Parent Loop BB0_6 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_8 Depth 3
                                        #         Child Loop BB0_10 Depth 4
                                        #           Child Loop BB0_12 Depth 5
                                        #             Child Loop BB0_13 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	.loc	1 0 5                           # src/matmul_seq_ikj.c:0:5
	movq	%r12, 56(%rsp)                  # 8-byte Spill
.Ltmp31:
	.loc	1 51 12 is_stmt 1               # src/matmul_seq_ikj.c:51:12
	shlq	$6, %r12
	movl	$4999, %r13d                    # imm = 0x1387
	subq	%r12, %r13
	cmpq	$64, %r13
	movl	$63, %eax
	cmovgeq	%rax, %r13
	incq	%r13
	movq	%r13, %rbp
	shrq	$2, %rbp
	decq	%rbp
	movq	%rsi, 64(%rsp)                  # 8-byte Spill
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 24(%rsp)                  # 8-byte Spill
	xorl	%r14d, %r14d
	jmp	.LBB0_8
.Ltmp32:
	.p2align	4, 0x90
.LBB0_16:                               #   in Loop: Header=BB0_8 Depth=3
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	#DEBUG_VALUE: i <- 0
	.loc	1 74 9                          # src/matmul_seq_ikj.c:74:9
	addq	$512, 24(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x200
	addq	$512, %rsi                      # imm = 0x200
.Ltmp33:
	.loc	1 74 27 is_stmt 0               # src/matmul_seq_ikj.c:74:27
	cmpq	$78, %r14
	leaq	1(%r14), %r14
.Ltmp34:
	.loc	1 74 9                          # src/matmul_seq_ikj.c:74:9
	je	.LBB0_17
.Ltmp35:
.LBB0_8:                                #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_10 Depth 4
                                        #           Child Loop BB0_12 Depth 5
                                        #             Child Loop BB0_13 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	.loc	1 51 12 is_stmt 1               # src/matmul_seq_ikj.c:51:12
	movq	%r14, %rax
	shlq	$6, %rax
	movl	$4999, %edx                     # imm = 0x1387
	subq	%rax, %rdx
	cmpq	$64, %rdx
	movl	$63, %eax
	cmovgeq	%rax, %rdx
	cmpq	$0, 72(%rsp)                    # 8-byte Folded Reload
	je	.LBB0_16
.Ltmp36:
# %bb.9:                                #   in Loop: Header=BB0_8 Depth=3
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	.loc	1 0 12 is_stmt 0                # src/matmul_seq_ikj.c:0:12
	movq	24(%rsp), %rax                  # 8-byte Reload
	xorl	%ecx, %ecx
	jmp	.LBB0_10
.Ltmp37:
	.p2align	4, 0x90
.LBB0_15:                               #   in Loop: Header=BB0_10 Depth=4
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	#DEBUG_VALUE: i <- 0
	.loc	1 74 9 is_stmt 1                # src/matmul_seq_ikj.c:74:9
	addq	$160000, %rax                   # imm = 0x27100
.Ltmp38:
	.loc	1 74 27 is_stmt 0               # src/matmul_seq_ikj.c:74:27
	cmpq	%r9, %rcx
	leaq	1(%rcx), %rcx
.Ltmp39:
	.loc	1 74 9                          # src/matmul_seq_ikj.c:74:9
	je	.LBB0_16
.Ltmp40:
.LBB0_10:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_8 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_12 Depth 5
                                        #             Child Loop BB0_13 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	#DEBUG_VALUE: i <- 0
	.loc	1 0 9                           # src/matmul_seq_ikj.c:0:9
	testq	%r13, %r13
	je	.LBB0_15
.Ltmp41:
# %bb.11:                               #   in Loop: Header=BB0_10 Depth=4
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	#DEBUG_VALUE: i <- 0
	.loc	1 76 28 is_stmt 1               # src/matmul_seq_ikj.c:76:28
	leaq	(%rdi,%rcx,4), %r8
	.loc	1 76 28 is_stmt 0 discriminator 2 # src/matmul_seq_ikj.c:76:28
	imulq	$40000, %r8, %r11               # imm = 0x9C40
	.loc	1 76 28 discriminator 4         # src/matmul_seq_ikj.c:76:28
	addq	%r15, %r11
	movq	%rsi, %r10
	xorl	%r8d, %r8d
.Ltmp42:
	.p2align	4, 0x90
.LBB0_12:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_8 Depth=3
                                        #         Parent Loop BB0_10 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_13 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	#DEBUG_VALUE: i <- 0
	.loc	1 76 28                         # src/matmul_seq_ikj.c:76:28
	leaq	(%r12,%r8,4), %rbx
	.loc	1 76 36                         # src/matmul_seq_ikj.c:76:36
	vbroadcastsd	(%r11,%rbx,8), %ymm0
	vmovups	%ymm0, 240(%rsp)                # 32-byte Spill
	vbroadcastsd	40000(%r11,%rbx,8), %ymm0
	vmovups	%ymm0, 208(%rsp)                # 32-byte Spill
	vbroadcastsd	80000(%r11,%rbx,8), %ymm0
	vmovups	%ymm0, 176(%rsp)                # 32-byte Spill
	vbroadcastsd	120000(%r11,%rbx,8), %ymm0
	vmovups	%ymm0, 144(%rsp)                # 32-byte Spill
	vbroadcastsd	8(%r11,%rbx,8), %ymm0
	vmovups	%ymm0, 112(%rsp)                # 32-byte Spill
	vbroadcastsd	40008(%r11,%rbx,8), %ymm5
	vbroadcastsd	80008(%r11,%rbx,8), %ymm6
	vbroadcastsd	120008(%r11,%rbx,8), %ymm7
	vbroadcastsd	16(%r11,%rbx,8), %ymm8
	vbroadcastsd	40016(%r11,%rbx,8), %ymm9
	vbroadcastsd	80016(%r11,%rbx,8), %ymm10
	vbroadcastsd	120016(%r11,%rbx,8), %ymm11
	vbroadcastsd	24(%r11,%rbx,8), %ymm12
	vbroadcastsd	40024(%r11,%rbx,8), %ymm13
	vbroadcastsd	80024(%r11,%rbx,8), %ymm14
	vbroadcastsd	120024(%r11,%rbx,8), %ymm15
	xorl	%ebx, %ebx
.Ltmp43:
	.p2align	4, 0x90
.LBB0_13:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_8 Depth=3
                                        #         Parent Loop BB0_10 Depth=4
                                        #           Parent Loop BB0_12 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	#DEBUG_VALUE: i <- 0
	.loc	1 76 38                         # src/matmul_seq_ikj.c:76:38
	vmovupd	-120000(%r10,%rbx,8), %ymm0     # AlignMOV convert to UnAlignMOV 
	vmovupd	-120000(%rax,%rbx,8), %ymm1     # AlignMOV convert to UnAlignMOV 
	.loc	1 76 25                         # src/matmul_seq_ikj.c:76:25
	vfmadd231pd	240(%rsp), %ymm0, %ymm1 # 32-byte Folded Reload
                                        # ymm1 = (ymm0 * mem) + ymm1
	vmovupd	-80000(%rax,%rbx,8), %ymm2      # AlignMOV convert to UnAlignMOV 
	vfmadd231pd	208(%rsp), %ymm0, %ymm2 # 32-byte Folded Reload
                                        # ymm2 = (ymm0 * mem) + ymm2
	vmovupd	-40000(%rax,%rbx,8), %ymm3      # AlignMOV convert to UnAlignMOV 
	vfmadd231pd	176(%rsp), %ymm0, %ymm3 # 32-byte Folded Reload
                                        # ymm3 = (ymm0 * mem) + ymm3
	vmovupd	144(%rsp), %ymm4                # 32-byte Reload
	vfmadd213pd	(%rax,%rbx,8), %ymm4, %ymm0 # ymm0 = (ymm4 * ymm0) + mem
	.loc	1 76 38                         # src/matmul_seq_ikj.c:76:38
	vmovupd	-80000(%r10,%rbx,8), %ymm4      # AlignMOV convert to UnAlignMOV 
	.loc	1 76 25                         # src/matmul_seq_ikj.c:76:25
	vfmadd231pd	112(%rsp), %ymm4, %ymm1 # 32-byte Folded Reload
                                        # ymm1 = (ymm4 * mem) + ymm1
	vfmadd231pd	%ymm5, %ymm4, %ymm2     # ymm2 = (ymm4 * ymm5) + ymm2
	vfmadd231pd	%ymm6, %ymm4, %ymm3     # ymm3 = (ymm4 * ymm6) + ymm3
	vfmadd231pd	%ymm4, %ymm7, %ymm0     # ymm0 = (ymm7 * ymm4) + ymm0
	.loc	1 76 38                         # src/matmul_seq_ikj.c:76:38
	vmovupd	-40000(%r10,%rbx,8), %ymm4      # AlignMOV convert to UnAlignMOV 
	.loc	1 76 25                         # src/matmul_seq_ikj.c:76:25
	vfmadd231pd	%ymm8, %ymm4, %ymm1     # ymm1 = (ymm4 * ymm8) + ymm1
	vfmadd231pd	%ymm9, %ymm4, %ymm2     # ymm2 = (ymm4 * ymm9) + ymm2
	vfmadd231pd	%ymm10, %ymm4, %ymm3    # ymm3 = (ymm4 * ymm10) + ymm3
	vfmadd231pd	%ymm4, %ymm11, %ymm0    # ymm0 = (ymm11 * ymm4) + ymm0
	.loc	1 76 38                         # src/matmul_seq_ikj.c:76:38
	vmovupd	(%r10,%rbx,8), %ymm4            # AlignMOV convert to UnAlignMOV 
	.loc	1 76 25                         # src/matmul_seq_ikj.c:76:25
	vfmadd231pd	%ymm12, %ymm4, %ymm1    # ymm1 = (ymm4 * ymm12) + ymm1
	vmovupd	%ymm1, -120000(%rax,%rbx,8)     # AlignMOV convert to UnAlignMOV 
	vfmadd231pd	%ymm13, %ymm4, %ymm2    # ymm2 = (ymm4 * ymm13) + ymm2
	vmovupd	%ymm2, -80000(%rax,%rbx,8)      # AlignMOV convert to UnAlignMOV 
	vfmadd231pd	%ymm14, %ymm4, %ymm3    # ymm3 = (ymm4 * ymm14) + ymm3
	vmovupd	%ymm3, -40000(%rax,%rbx,8)      # AlignMOV convert to UnAlignMOV 
	vfmadd231pd	%ymm4, %ymm15, %ymm0    # ymm0 = (ymm15 * ymm4) + ymm0
	vmovupd	%ymm0, (%rax,%rbx,8)            # AlignMOV convert to UnAlignMOV 
	.loc	1 75 31 is_stmt 1               # src/matmul_seq_ikj.c:75:31
	addq	$4, %rbx
	cmpq	%rdx, %rbx
.Ltmp44:
	.loc	1 75 13 is_stmt 0               # src/matmul_seq_ikj.c:75:13
	jle	.LBB0_13
.Ltmp45:
# %bb.14:                               #   in Loop: Header=BB0_12 Depth=5
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	#DEBUG_VALUE: i <- 0
	addq	$160000, %r10                   # imm = 0x27100
.Ltmp46:
	.loc	1 75 31                         # src/matmul_seq_ikj.c:75:31
	cmpq	%rbp, %r8
	leaq	1(%r8), %r8
.Ltmp47:
	.loc	1 75 13                         # src/matmul_seq_ikj.c:75:13
	jne	.LBB0_12
	jmp	.LBB0_15
.Ltmp48:
.LBB0_21:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $rax
	.loc	1 52 9 is_stmt 1                # src/matmul_seq_ikj.c:52:9
	movl	$.L.str, %edi
.Ltmp49:
.LBB0_22:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	.loc	1 0 0 is_stmt 0                 # src/matmul_seq_ikj.c:0:0
	callq	perror
.Ltmp50:
	movl	$1, %eax
	jmp	.LBB0_23
.Ltmp51:
.LBB0_19:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	leaq	80(%rsp), %rsi
	.loc	1 81 9 is_stmt 1                # src/matmul_seq_ikj.c:81:9
	movl	$1, %edi
	vzeroupper
	callq	clock_gettime
.Ltmp52:
	.loc	1 82 34                         # src/matmul_seq_ikj.c:82:34
	movq	80(%rsp), %rax
	.loc	1 82 64 is_stmt 0               # src/matmul_seq_ikj.c:82:64
	movq	88(%rsp), %rcx
	.loc	1 82 41                         # src/matmul_seq_ikj.c:82:41
	subq	96(%rsp), %rax
	.loc	1 82 29                         # src/matmul_seq_ikj.c:82:29
	vxorps	%xmm5, %xmm5, %xmm5
	vcvtsi2sd	%rax, %xmm5, %xmm1
	.loc	1 82 72                         # src/matmul_seq_ikj.c:82:72
	subq	104(%rsp), %rcx
	.loc	1 82 59                         # src/matmul_seq_ikj.c:82:59
	vxorps	%xmm5, %xmm5, %xmm5
	vcvtsi2sd	%rcx, %xmm5, %xmm0
	.loc	1 82 57                         # src/matmul_seq_ikj.c:82:57
	vfmadd132sd	.LCPI0_2(%rip), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
.Ltmp53:
	#DEBUG_VALUE: main:time_taken <- $xmm0
	.loc	1 84 17 is_stmt 1               # src/matmul_seq_ikj.c:84:17
	movq	stderr(%rip), %rdi
	.loc	1 84 9 is_stmt 0                # src/matmul_seq_ikj.c:84:9
	movl	$.L.str.1, %esi
	movl	$5000, %edx                     # imm = 0x1388
	movb	$1, %al
	callq	fprintf
.Ltmp54:
	.loc	1 89 15 is_stmt 1               # src/matmul_seq_ikj.c:89:15
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %esi
	callq	fopen
.Ltmp55:
	#DEBUG_VALUE: main:f <- $rax
	.loc	1 90 10                         # src/matmul_seq_ikj.c:90:10
	testq	%rax, %rax
.Ltmp56:
	.loc	1 90 9 is_stmt 0                # src/matmul_seq_ikj.c:90:9
	je	.LBB0_20
.Ltmp57:
# %bb.24:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $rax
	.loc	1 89 15 is_stmt 1               # src/matmul_seq_ikj.c:89:15
	movq	%rax, %r12
	xorl	%r14d, %r14d
	.loc	1 96 5                          # src/matmul_seq_ikj.c:96:5
	movl	$.L.str.5, %esi
	movq	%rax, %rdi
	movl	$5000, %edx                     # imm = 0x1388
	xorl	%eax, %eax
.Ltmp58:
	#DEBUG_VALUE: main:f <- $r12
	callq	fprintf
.Ltmp59:
	#DEBUG_VALUE: i <- 0
	.loc	1 0 5 is_stmt 0                 # src/matmul_seq_ikj.c:0:5
	movq	8(%rsp), %rbp                   # 8-byte Reload
	addq	$56, %rbp
.Ltmp60:
	.p2align	4, 0x90
.LBB0_25:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_26 Depth 2
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	#DEBUG_VALUE: i <- 0
	xorl	%r13d, %r13d
.Ltmp61:
	.p2align	4, 0x90
.LBB0_26:                               #   Parent Loop BB0_25 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	#DEBUG_VALUE: i <- 0
	.loc	1 101 33 is_stmt 1              # src/matmul_seq_ikj.c:101:33
	vmovsd	-56(%rbp,%r13), %xmm0           # xmm0 = mem[0],zero
	.loc	1 101 13 is_stmt 0              # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp62:
	.loc	1 101 33                        # src/matmul_seq_ikj.c:101:33
	vmovsd	-48(%rbp,%r13), %xmm0           # xmm0 = mem[0],zero
	.loc	1 101 13                        # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp63:
	.loc	1 101 33                        # src/matmul_seq_ikj.c:101:33
	vmovsd	-40(%rbp,%r13), %xmm0           # xmm0 = mem[0],zero
	.loc	1 101 13                        # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp64:
	.loc	1 101 33                        # src/matmul_seq_ikj.c:101:33
	vmovsd	-32(%rbp,%r13), %xmm0           # xmm0 = mem[0],zero
	.loc	1 101 13                        # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp65:
	.loc	1 101 33                        # src/matmul_seq_ikj.c:101:33
	vmovsd	-24(%rbp,%r13), %xmm0           # xmm0 = mem[0],zero
	.loc	1 101 13                        # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp66:
	.loc	1 101 33                        # src/matmul_seq_ikj.c:101:33
	vmovsd	-16(%rbp,%r13), %xmm0           # xmm0 = mem[0],zero
	.loc	1 101 13                        # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp67:
	.loc	1 101 33                        # src/matmul_seq_ikj.c:101:33
	vmovsd	-8(%rbp,%r13), %xmm0            # xmm0 = mem[0],zero
	.loc	1 101 13                        # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp68:
	.loc	1 101 33                        # src/matmul_seq_ikj.c:101:33
	vmovsd	(%rbp,%r13), %xmm0              # xmm0 = mem[0],zero
	.loc	1 101 13                        # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp69:
	.loc	1 99 27 is_stmt 1               # src/matmul_seq_ikj.c:99:27
	addq	$64, %r13
	cmpq	$8000, %r13                     # imm = 0x1F40
.Ltmp70:
	.loc	1 99 9 is_stmt 0                # src/matmul_seq_ikj.c:99:9
	jne	.LBB0_26
.Ltmp71:
# %bb.27:                               #   in Loop: Header=BB0_25 Depth=1
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	#DEBUG_VALUE: i <- 0
	.loc	1 103 9 is_stmt 1               # src/matmul_seq_ikj.c:103:9
	movl	$10, %edi
	movq	%r12, %rsi
	callq	fputc@PLT
.Ltmp72:
	.loc	1 97 5                          # src/matmul_seq_ikj.c:97:5
	addq	$40000, %rbp                    # imm = 0x9C40
.Ltmp73:
	.loc	1 97 23 is_stmt 0               # src/matmul_seq_ikj.c:97:23
	cmpq	$999, %r14                      # imm = 0x3E7
	leaq	1(%r14), %r14
.Ltmp74:
	.loc	1 97 5                          # src/matmul_seq_ikj.c:97:5
	jne	.LBB0_25
.Ltmp75:
# %bb.28:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	.loc	1 106 5 is_stmt 1               # src/matmul_seq_ikj.c:106:5
	movq	%r12, %rdi
	callq	fclose
.Ltmp76:
	.loc	1 109 5                         # src/matmul_seq_ikj.c:109:5
	movq	%r15, %rdi
	callq	free
.Ltmp77:
	.loc	1 0 5 is_stmt 0                 # src/matmul_seq_ikj.c:0:5
	movq	32(%rsp), %rdi                  # 8-byte Reload
	.loc	1 110 5 is_stmt 1               # src/matmul_seq_ikj.c:110:5
	callq	free
.Ltmp78:
	.loc	1 0 5 is_stmt 0                 # src/matmul_seq_ikj.c:0:5
	movq	8(%rsp), %rdi                   # 8-byte Reload
	.loc	1 111 5 is_stmt 1               # src/matmul_seq_ikj.c:111:5
	callq	free
.Ltmp79:
	.loc	1 0 5 is_stmt 0                 # src/matmul_seq_ikj.c:0:5
	xorl	%eax, %eax
.Ltmp80:
.LBB0_23:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	.loc	1 113 1 epilogue_begin is_stmt 1 # src/matmul_seq_ikj.c:113:1
	addq	$280, %rsp                      # imm = 0x118
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
.Ltmp81:
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Ltmp82:
.LBB0_20:
	.cfi_def_cfa_offset 336
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $r15
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $rax
	.loc	1 92 9                          # src/matmul_seq_ikj.c:92:9
	movl	$.L.str.4, %edi
	jmp	.LBB0_22
.Ltmp83:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
	.file	2 "/usr/include" "stdlib.h"
	.file	3 "/opt/intel/oneapi/compiler/2023.2.1/linux/lib/clang/17/include" "stddef.h"
	.file	4 "/usr/include" "time.h"
	.file	5 "/usr/include/x86_64-linux-gnu/bits" "types.h"
	.file	6 "/usr/include/x86_64-linux-gnu/bits/types" "clockid_t.h"
	.file	7 "/usr/include/x86_64-linux-gnu/bits/types" "struct_timespec.h"
	.file	8 "/usr/include" "stdio.h"
	.file	9 "/usr/include/x86_64-linux-gnu/bits/types" "struct_FILE.h"
	.file	10 "/usr/include/x86_64-linux-gnu/bits/types" "FILE.h"
                                        # -- End function
	.type	.L.str,@object                  # 
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Memory allocation failed"
	.size	.L.str, 25

	.type	.L.str.1,@object                # 
.L.str.1:
	.asciz	"[seq-ikj] N=%d, elapsed=%.3f s\n"
	.size	.L.str.1, 32

	.type	.L.str.2,@object                # 
.L.str.2:
	.asciz	"mat-res.txt"
	.size	.L.str.2, 12

	.type	.L.str.3,@object                # 
.L.str.3:
	.asciz	"w"
	.size	.L.str.3, 2

	.type	.L.str.4,@object                # 
.L.str.4:
	.asciz	"fopen"
	.size	.L.str.4, 6

	.type	.L.str.5,@object                # 
.L.str.5:
	.asciz	"%d\n\n"
	.size	.L.str.5, 5

	.type	.L.str.6,@object                # 
.L.str.6:
	.asciz	"%.0f "
	.size	.L.str.6, 6

	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Ltmp1-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	85                              # super-register DW_OP_reg5
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	85                              # super-register DW_OP_reg5
	.byte	159                             # DW_OP_stack_value
	.quad	0
	.quad	0
.Ldebug_loc1:
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Ltmp0-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	84                              # DW_OP_reg4
	.quad	.Ltmp0-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	84                              # DW_OP_reg4
	.byte	159                             # DW_OP_stack_value
	.quad	0
	.quad	0
.Ldebug_loc2:
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	95                              # DW_OP_reg15
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	95                              # DW_OP_reg15
	.quad	0
	.quad	0
.Ldebug_loc3:
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	94                              # DW_OP_reg14
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.short	2                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	32                              # 32
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	94                              # DW_OP_reg14
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.short	2                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	32                              # 32
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	2                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	32                              # 32
	.quad	0
	.quad	0
.Ldebug_loc4:
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	80                              # DW_OP_reg0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	92                              # DW_OP_reg12
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.short	2                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	8                               # 8
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	80                              # DW_OP_reg0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.short	2                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	8                               # 8
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	2                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	8                               # 8
	.quad	0
	.quad	0
.Ldebug_loc5:
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	17                              # DW_OP_consts
	.byte	0                               # 0
	.byte	159                             # DW_OP_stack_value
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	17                              # DW_OP_consts
	.byte	0                               # 0
	.byte	159                             # DW_OP_stack_value
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	17                              # DW_OP_consts
	.byte	0                               # 0
	.byte	159                             # DW_OP_stack_value
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	17                              # DW_OP_consts
	.byte	0                               # 0
	.byte	159                             # DW_OP_stack_value
	.quad	0
	.quad	0
.Ldebug_loc6:
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	97                              # DW_OP_reg17
	.quad	0
	.quad	0
.Ldebug_loc7:
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	80                              # DW_OP_reg0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	92                              # DW_OP_reg12
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	80                              # DW_OP_reg0
	.quad	0
	.quad	0
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.ascii	"\201v"                         # DW_AT_INTEL_comp_flags
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	55                              # DW_AT_count
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\227B"                         # DW_AT_GNU_all_call_sites
	.byte	25                              # DW_FORM_flag_present
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	23                              # DW_FORM_sec_offset
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	23                              # DW_FORM_sec_offset
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.byte	11                              # DW_TAG_lexical_block
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	14                              # Abbreviation Code
	.byte	11                              # DW_TAG_lexical_block
	.byte	1                               # DW_CHILDREN_yes
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	15                              # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	16                              # Abbreviation Code
	.ascii	"\211\202\001"                  # DW_TAG_GNU_call_site
	.byte	1                               # DW_CHILDREN_yes
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	17                              # Abbreviation Code
	.ascii	"\212\202\001"                  # DW_TAG_GNU_call_site_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\221B"                         # DW_AT_GNU_call_site_value
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	18                              # Abbreviation Code
	.ascii	"\211\202\001"                  # DW_TAG_GNU_call_site
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	19                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	20                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	21                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	22                              # Abbreviation Code
	.byte	22                              # DW_TAG_typedef
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	23                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	24                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	25                              # Abbreviation Code
	.byte	13                              # DW_TAG_member
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	56                              # DW_AT_data_member_location
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	26                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	5                               # DW_FORM_data2
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	27                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	28                              # Abbreviation Code
	.byte	24                              # DW_TAG_unspecified_parameters
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	29                              # Abbreviation Code
	.byte	55                              # DW_TAG_restrict_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	30                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	31                              # Abbreviation Code
	.byte	22                              # DW_TAG_typedef
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	32                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	33                              # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	55                              # DW_AT_count
	.byte	5                               # DW_FORM_data2
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x719 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.long	.Linfo_string1                  # DW_AT_INTEL_comp_flags
	.short	29                              # DW_AT_language
	.long	.Linfo_string2                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string3                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2e:0x11 DW_TAG_variable
	.long	63                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	52                              # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	.L.str
	.byte	3                               # Abbrev [3] 0x3f:0xc DW_TAG_array_type
	.long	75                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x44:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	25                              # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x4b:0x7 DW_TAG_base_type
	.long	.Linfo_string4                  # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	6                               # Abbrev [6] 0x52:0x7 DW_TAG_base_type
	.long	.Linfo_string5                  # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # DW_AT_encoding
	.byte	2                               # Abbrev [2] 0x59:0x11 DW_TAG_variable
	.long	106                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	84                              # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	.L.str.1
	.byte	3                               # Abbrev [3] 0x6a:0xc DW_TAG_array_type
	.long	75                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x6f:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	32                              # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	2                               # Abbrev [2] 0x76:0x11 DW_TAG_variable
	.long	135                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	89                              # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	.L.str.2
	.byte	3                               # Abbrev [3] 0x87:0xc DW_TAG_array_type
	.long	75                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x8c:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	12                              # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	2                               # Abbrev [2] 0x93:0x11 DW_TAG_variable
	.long	164                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	89                              # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	.L.str.3
	.byte	3                               # Abbrev [3] 0xa4:0xc DW_TAG_array_type
	.long	75                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0xa9:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	2                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	2                               # Abbrev [2] 0xb0:0x11 DW_TAG_variable
	.long	193                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	92                              # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	.L.str.4
	.byte	3                               # Abbrev [3] 0xc1:0xc DW_TAG_array_type
	.long	75                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0xc6:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	6                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	2                               # Abbrev [2] 0xcd:0x11 DW_TAG_variable
	.long	222                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	96                              # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	.L.str.5
	.byte	3                               # Abbrev [3] 0xde:0xc DW_TAG_array_type
	.long	75                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0xe3:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	5                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	2                               # Abbrev [2] 0xea:0x11 DW_TAG_variable
	.long	193                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	101                             # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	.L.str.6
	.byte	7                               # Abbrev [7] 0xfb:0x7 DW_TAG_variable
	.long	164                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	103                             # DW_AT_decl_line
	.byte	8                               # Abbrev [8] 0x102:0x308 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string63                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	38                              # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1099                            # DW_AT_type
                                        # DW_AT_external
	.byte	9                               # Abbrev [9] 0x11b:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc0                    # DW_AT_location
	.long	.Linfo_string66                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	38                              # DW_AT_decl_line
	.long	1099                            # DW_AT_type
	.byte	9                               # Abbrev [9] 0x12a:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc1                    # DW_AT_location
	.long	.Linfo_string67                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	38                              # DW_AT_decl_line
	.long	1792                            # DW_AT_type
	.byte	10                              # Abbrev [10] 0x139:0xf DW_TAG_variable
	.byte	3                               # DW_AT_location
	.byte	145
	.asciz	"\340"
	.long	.Linfo_string64                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	68                              # DW_AT_decl_line
	.long	1133                            # DW_AT_type
	.byte	10                              # Abbrev [10] 0x148:0xf DW_TAG_variable
	.byte	3                               # DW_AT_location
	.byte	145
	.asciz	"\320"
	.long	.Linfo_string65                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	68                              # DW_AT_decl_line
	.long	1133                            # DW_AT_type
	.byte	11                              # Abbrev [11] 0x157:0xf DW_TAG_variable
	.long	.Ldebug_loc2                    # DW_AT_location
	.long	.Linfo_string68                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	47                              # DW_AT_decl_line
	.long	1797                            # DW_AT_type
	.byte	11                              # Abbrev [11] 0x166:0xf DW_TAG_variable
	.long	.Ldebug_loc3                    # DW_AT_location
	.long	.Linfo_string70                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	48                              # DW_AT_decl_line
	.long	1797                            # DW_AT_type
	.byte	11                              # Abbrev [11] 0x175:0xf DW_TAG_variable
	.long	.Ldebug_loc4                    # DW_AT_location
	.long	.Linfo_string71                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
	.long	1797                            # DW_AT_type
	.byte	11                              # Abbrev [11] 0x184:0xf DW_TAG_variable
	.long	.Ldebug_loc6                    # DW_AT_location
	.long	.Linfo_string73                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	82                              # DW_AT_decl_line
	.long	1820                            # DW_AT_type
	.byte	11                              # Abbrev [11] 0x193:0xf DW_TAG_variable
	.long	.Ldebug_loc7                    # DW_AT_location
	.long	.Linfo_string74                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	89                              # DW_AT_decl_line
	.long	1248                            # DW_AT_type
	.byte	12                              # Abbrev [12] 0x1a2:0x32 DW_TAG_lexical_block
	.quad	.Ltmp11                         # DW_AT_low_pc
	.long	.Ltmp16-.Ltmp11                 # DW_AT_high_pc
	.byte	13                              # Abbrev [13] 0x1af:0xb DW_TAG_variable
	.long	.Linfo_string72                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.long	1099                            # DW_AT_type
	.byte	12                              # Abbrev [12] 0x1ba:0x19 DW_TAG_lexical_block
	.quad	.Ltmp11                         # DW_AT_low_pc
	.long	.Ltmp16-.Ltmp11                 # DW_AT_high_pc
	.byte	13                              # Abbrev [13] 0x1c7:0xb DW_TAG_variable
	.long	.Linfo_string75                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
	.long	1099                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	14                              # Abbrev [14] 0x1d4:0x3f DW_TAG_lexical_block
	.long	.Ldebug_ranges0                 # DW_AT_ranges
	.byte	11                              # Abbrev [11] 0x1d9:0xf DW_TAG_variable
	.long	.Ldebug_loc5                    # DW_AT_location
	.long	.Linfo_string72                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.long	1099                            # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1e8:0x2a DW_TAG_lexical_block
	.long	.Ldebug_ranges1                 # DW_AT_ranges
	.byte	13                              # Abbrev [13] 0x1ed:0xb DW_TAG_variable
	.long	.Linfo_string76                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	74                              # DW_AT_decl_line
	.long	1099                            # DW_AT_type
	.byte	12                              # Abbrev [12] 0x1f8:0x19 DW_TAG_lexical_block
	.quad	.Ltmp41                         # DW_AT_low_pc
	.long	.Ltmp48-.Ltmp41                 # DW_AT_high_pc
	.byte	13                              # Abbrev [13] 0x205:0xb DW_TAG_variable
	.long	.Linfo_string75                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
	.long	1099                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x213:0x33 DW_TAG_lexical_block
	.quad	.Ltmp61                         # DW_AT_low_pc
	.long	.Ltmp75-.Ltmp61                 # DW_AT_high_pc
	.byte	15                              # Abbrev [15] 0x220:0xc DW_TAG_variable
	.byte	0                               # DW_AT_const_value
	.long	.Linfo_string72                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	97                              # DW_AT_decl_line
	.long	1099                            # DW_AT_type
	.byte	12                              # Abbrev [12] 0x22c:0x19 DW_TAG_lexical_block
	.quad	.Ltmp61                         # DW_AT_low_pc
	.long	.Ltmp71-.Ltmp61                 # DW_AT_high_pc
	.byte	13                              # Abbrev [13] 0x239:0xb DW_TAG_variable
	.long	.Linfo_string75                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	99                              # DW_AT_decl_line
	.long	1099                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x246:0x1d DW_TAG_GNU_call_site
	.long	1034                            # DW_AT_abstract_origin
	.quad	.Ltmp3                          # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x253:0x9 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	5                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\200\204\257_"
	.byte	17                              # Abbrev [17] 0x25c:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	16
	.byte	64
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x263:0x1d DW_TAG_GNU_call_site
	.long	1034                            # DW_AT_abstract_origin
	.quad	.Ltmp5                          # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x270:0x9 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	5                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\200\204\257_"
	.byte	17                              # Abbrev [17] 0x279:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	16
	.byte	64
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x280:0x1d DW_TAG_GNU_call_site
	.long	1034                            # DW_AT_abstract_origin
	.quad	.Ltmp7                          # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x28d:0x9 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	5                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\200\204\257_"
	.byte	17                              # Abbrev [17] 0x296:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	16
	.byte	64
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x29d:0x1a DW_TAG_GNU_call_site
	.long	1076                            # DW_AT_abstract_origin
	.quad	.Ltmp18                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x2aa:0x5 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	1                               # DW_AT_GNU_call_site_value
	.byte	49
	.byte	17                              # Abbrev [17] 0x2af:0x7 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	3                               # DW_AT_GNU_call_site_value
	.byte	145
	.asciz	"\340"
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0x2b7:0xd DW_TAG_GNU_call_site
	.long	1195                            # DW_AT_abstract_origin
	.quad	.Ltmp50                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x2c4:0x1a DW_TAG_GNU_call_site
	.long	1076                            # DW_AT_abstract_origin
	.quad	.Ltmp52                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x2d1:0x5 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	1                               # DW_AT_GNU_call_site_value
	.byte	49
	.byte	17                              # Abbrev [17] 0x2d6:0x7 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	3                               # DW_AT_GNU_call_site_value
	.byte	145
	.asciz	"\320"
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x2de:0x15 DW_TAG_GNU_call_site
	.long	1219                            # DW_AT_abstract_origin
	.quad	.Ltmp54                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x2eb:0x7 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	81
	.byte	3                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\210'"
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0x2f3:0xd DW_TAG_GNU_call_site
	.long	1738                            # DW_AT_abstract_origin
	.quad	.Ltmp55                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x300:0x15 DW_TAG_GNU_call_site
	.long	1219                            # DW_AT_abstract_origin
	.quad	.Ltmp59                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x30d:0x7 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	81
	.byte	3                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\210'"
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x315:0x14 DW_TAG_GNU_call_site
	.long	1219                            # DW_AT_abstract_origin
	.quad	.Ltmp62                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x322:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x329:0x14 DW_TAG_GNU_call_site
	.long	1219                            # DW_AT_abstract_origin
	.quad	.Ltmp63                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x336:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x33d:0x14 DW_TAG_GNU_call_site
	.long	1219                            # DW_AT_abstract_origin
	.quad	.Ltmp64                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x34a:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x351:0x14 DW_TAG_GNU_call_site
	.long	1219                            # DW_AT_abstract_origin
	.quad	.Ltmp65                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x35e:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x365:0x14 DW_TAG_GNU_call_site
	.long	1219                            # DW_AT_abstract_origin
	.quad	.Ltmp66                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x372:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x379:0x14 DW_TAG_GNU_call_site
	.long	1219                            # DW_AT_abstract_origin
	.quad	.Ltmp67                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x386:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x38d:0x14 DW_TAG_GNU_call_site
	.long	1219                            # DW_AT_abstract_origin
	.quad	.Ltmp68                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x39a:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x3a1:0x14 DW_TAG_GNU_call_site
	.long	1219                            # DW_AT_abstract_origin
	.quad	.Ltmp69                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x3ae:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x3b5:0x14 DW_TAG_GNU_call_site
	.long	1761                            # DW_AT_abstract_origin
	.quad	.Ltmp76                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x3c2:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x3c9:0x14 DW_TAG_GNU_call_site
	.long	1778                            # DW_AT_abstract_origin
	.quad	.Ltmp77                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x3d6:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	127
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x3dd:0x16 DW_TAG_GNU_call_site
	.long	1778                            # DW_AT_abstract_origin
	.quad	.Ltmp78                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x3ea:0x8 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	4                               # DW_AT_GNU_call_site_value
	.byte	145
	.byte	32
	.byte	148
	.byte	8
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x3f3:0x16 DW_TAG_GNU_call_site
	.long	1778                            # DW_AT_abstract_origin
	.quad	.Ltmp79                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x400:0x8 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	4                               # DW_AT_GNU_call_site_value
	.byte	145
	.byte	8
	.byte	148
	.byte	8
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x40a:0x17 DW_TAG_subprogram
	.long	.Linfo_string6                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	605                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1057                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	20                              # Abbrev [20] 0x416:0x5 DW_TAG_formal_parameter
	.long	1058                            # DW_AT_type
	.byte	20                              # Abbrev [20] 0x41b:0x5 DW_TAG_formal_parameter
	.long	1058                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x421:0x1 DW_TAG_pointer_type
	.byte	22                              # Abbrev [22] 0x422:0xb DW_TAG_typedef
	.long	1069                            # DW_AT_type
	.long	.Linfo_string8                  # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
	.byte	5                               # Abbrev [5] 0x42d:0x7 DW_TAG_base_type
	.long	.Linfo_string7                  # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	19                              # Abbrev [19] 0x434:0x17 DW_TAG_subprogram
	.long	.Linfo_string9                  # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.short	288                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1099                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	20                              # Abbrev [20] 0x440:0x5 DW_TAG_formal_parameter
	.long	1106                            # DW_AT_type
	.byte	20                              # Abbrev [20] 0x445:0x5 DW_TAG_formal_parameter
	.long	1128                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x44b:0x7 DW_TAG_base_type
	.long	.Linfo_string10                 # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	22                              # Abbrev [22] 0x452:0xb DW_TAG_typedef
	.long	1117                            # DW_AT_type
	.long	.Linfo_string12                 # DW_AT_name
	.byte	6                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x45d:0xb DW_TAG_typedef
	.long	1099                            # DW_AT_type
	.long	.Linfo_string11                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	169                             # DW_AT_decl_line
	.byte	23                              # Abbrev [23] 0x468:0x5 DW_TAG_pointer_type
	.long	1133                            # DW_AT_type
	.byte	24                              # Abbrev [24] 0x46d:0x21 DW_TAG_structure_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	16                              # DW_AT_byte_size
	.byte	7                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0x475:0xc DW_TAG_member
	.long	.Linfo_string13                 # DW_AT_name
	.long	1166                            # DW_AT_type
	.byte	7                               # DW_AT_decl_file
	.byte	16                              # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x481:0xc DW_TAG_member
	.long	.Linfo_string16                 # DW_AT_name
	.long	1184                            # DW_AT_type
	.byte	7                               # DW_AT_decl_file
	.byte	21                              # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	22                              # Abbrev [22] 0x48e:0xb DW_TAG_typedef
	.long	1177                            # DW_AT_type
	.long	.Linfo_string15                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	160                             # DW_AT_decl_line
	.byte	5                               # Abbrev [5] 0x499:0x7 DW_TAG_base_type
	.long	.Linfo_string14                 # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	22                              # Abbrev [22] 0x4a0:0xb DW_TAG_typedef
	.long	1177                            # DW_AT_type
	.long	.Linfo_string17                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	197                             # DW_AT_decl_line
	.byte	26                              # Abbrev [26] 0x4ab:0xe DW_TAG_subprogram
	.long	.Linfo_string19                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.short	804                             # DW_AT_decl_line
                                        # DW_AT_prototyped
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	20                              # Abbrev [20] 0x4b3:0x5 DW_TAG_formal_parameter
	.long	1209                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x4b9:0x5 DW_TAG_pointer_type
	.long	1214                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x4be:0x5 DW_TAG_const_type
	.long	75                              # DW_AT_type
	.byte	19                              # Abbrev [19] 0x4c3:0x18 DW_TAG_subprogram
	.long	.Linfo_string20                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.short	350                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1099                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	20                              # Abbrev [20] 0x4cf:0x5 DW_TAG_formal_parameter
	.long	1243                            # DW_AT_type
	.byte	20                              # Abbrev [20] 0x4d4:0x5 DW_TAG_formal_parameter
	.long	1733                            # DW_AT_type
	.byte	28                              # Abbrev [28] 0x4d9:0x1 DW_TAG_unspecified_parameters
	.byte	0                               # End Of Children Mark
	.byte	29                              # Abbrev [29] 0x4db:0x5 DW_TAG_restrict_type
	.long	1248                            # DW_AT_type
	.byte	23                              # Abbrev [23] 0x4e0:0x5 DW_TAG_pointer_type
	.long	1253                            # DW_AT_type
	.byte	22                              # Abbrev [22] 0x4e5:0xb DW_TAG_typedef
	.long	1264                            # DW_AT_type
	.long	.Linfo_string59                 # DW_AT_name
	.byte	10                              # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x4f0:0x165 DW_TAG_structure_type
	.long	.Linfo_string58                 # DW_AT_name
	.byte	216                             # DW_AT_byte_size
	.byte	9                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0x4f8:0xc DW_TAG_member
	.long	.Linfo_string21                 # DW_AT_name
	.long	1099                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	51                              # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x504:0xc DW_TAG_member
	.long	.Linfo_string22                 # DW_AT_name
	.long	1621                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x510:0xc DW_TAG_member
	.long	.Linfo_string23                 # DW_AT_name
	.long	1621                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	55                              # DW_AT_decl_line
	.byte	16                              # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x51c:0xc DW_TAG_member
	.long	.Linfo_string24                 # DW_AT_name
	.long	1621                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	56                              # DW_AT_decl_line
	.byte	24                              # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x528:0xc DW_TAG_member
	.long	.Linfo_string25                 # DW_AT_name
	.long	1621                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	57                              # DW_AT_decl_line
	.byte	32                              # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x534:0xc DW_TAG_member
	.long	.Linfo_string26                 # DW_AT_name
	.long	1621                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.byte	40                              # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x540:0xc DW_TAG_member
	.long	.Linfo_string27                 # DW_AT_name
	.long	1621                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
	.byte	48                              # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x54c:0xc DW_TAG_member
	.long	.Linfo_string28                 # DW_AT_name
	.long	1621                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	60                              # DW_AT_decl_line
	.byte	56                              # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x558:0xc DW_TAG_member
	.long	.Linfo_string29                 # DW_AT_name
	.long	1621                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	61                              # DW_AT_decl_line
	.byte	64                              # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x564:0xc DW_TAG_member
	.long	.Linfo_string30                 # DW_AT_name
	.long	1621                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	64                              # DW_AT_decl_line
	.byte	72                              # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x570:0xc DW_TAG_member
	.long	.Linfo_string31                 # DW_AT_name
	.long	1621                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	65                              # DW_AT_decl_line
	.byte	80                              # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x57c:0xc DW_TAG_member
	.long	.Linfo_string32                 # DW_AT_name
	.long	1621                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	66                              # DW_AT_decl_line
	.byte	88                              # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x588:0xc DW_TAG_member
	.long	.Linfo_string33                 # DW_AT_name
	.long	1626                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	68                              # DW_AT_decl_line
	.byte	96                              # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x594:0xc DW_TAG_member
	.long	.Linfo_string35                 # DW_AT_name
	.long	1636                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	70                              # DW_AT_decl_line
	.byte	104                             # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x5a0:0xc DW_TAG_member
	.long	.Linfo_string36                 # DW_AT_name
	.long	1099                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	72                              # DW_AT_decl_line
	.byte	112                             # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x5ac:0xc DW_TAG_member
	.long	.Linfo_string37                 # DW_AT_name
	.long	1099                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.byte	116                             # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x5b8:0xc DW_TAG_member
	.long	.Linfo_string38                 # DW_AT_name
	.long	1641                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	74                              # DW_AT_decl_line
	.byte	120                             # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x5c4:0xc DW_TAG_member
	.long	.Linfo_string40                 # DW_AT_name
	.long	1652                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	77                              # DW_AT_decl_line
	.byte	128                             # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x5d0:0xc DW_TAG_member
	.long	.Linfo_string42                 # DW_AT_name
	.long	1659                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	78                              # DW_AT_decl_line
	.byte	130                             # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x5dc:0xc DW_TAG_member
	.long	.Linfo_string44                 # DW_AT_name
	.long	1666                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	79                              # DW_AT_decl_line
	.byte	131                             # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x5e8:0xc DW_TAG_member
	.long	.Linfo_string45                 # DW_AT_name
	.long	1678                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	81                              # DW_AT_decl_line
	.byte	136                             # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x5f4:0xc DW_TAG_member
	.long	.Linfo_string47                 # DW_AT_name
	.long	1690                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	89                              # DW_AT_decl_line
	.byte	144                             # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x600:0xc DW_TAG_member
	.long	.Linfo_string49                 # DW_AT_name
	.long	1701                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	91                              # DW_AT_decl_line
	.byte	152                             # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x60c:0xc DW_TAG_member
	.long	.Linfo_string51                 # DW_AT_name
	.long	1711                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	92                              # DW_AT_decl_line
	.byte	160                             # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x618:0xc DW_TAG_member
	.long	.Linfo_string53                 # DW_AT_name
	.long	1636                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	93                              # DW_AT_decl_line
	.byte	168                             # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x624:0xc DW_TAG_member
	.long	.Linfo_string54                 # DW_AT_name
	.long	1057                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	94                              # DW_AT_decl_line
	.byte	176                             # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x630:0xc DW_TAG_member
	.long	.Linfo_string55                 # DW_AT_name
	.long	1058                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	95                              # DW_AT_decl_line
	.byte	184                             # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x63c:0xc DW_TAG_member
	.long	.Linfo_string56                 # DW_AT_name
	.long	1099                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	96                              # DW_AT_decl_line
	.byte	192                             # DW_AT_data_member_location
	.byte	25                              # Abbrev [25] 0x648:0xc DW_TAG_member
	.long	.Linfo_string57                 # DW_AT_name
	.long	1721                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	98                              # DW_AT_decl_line
	.byte	196                             # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x655:0x5 DW_TAG_pointer_type
	.long	75                              # DW_AT_type
	.byte	23                              # Abbrev [23] 0x65a:0x5 DW_TAG_pointer_type
	.long	1631                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x65f:0x5 DW_TAG_structure_type
	.long	.Linfo_string34                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	23                              # Abbrev [23] 0x664:0x5 DW_TAG_pointer_type
	.long	1264                            # DW_AT_type
	.byte	22                              # Abbrev [22] 0x669:0xb DW_TAG_typedef
	.long	1177                            # DW_AT_type
	.long	.Linfo_string39                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	152                             # DW_AT_decl_line
	.byte	5                               # Abbrev [5] 0x674:0x7 DW_TAG_base_type
	.long	.Linfo_string41                 # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	2                               # DW_AT_byte_size
	.byte	5                               # Abbrev [5] 0x67b:0x7 DW_TAG_base_type
	.long	.Linfo_string43                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x682:0xc DW_TAG_array_type
	.long	75                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x687:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x68e:0x5 DW_TAG_pointer_type
	.long	1683                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0x693:0x7 DW_TAG_typedef
	.long	.Linfo_string46                 # DW_AT_name
	.byte	9                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x69a:0xb DW_TAG_typedef
	.long	1177                            # DW_AT_type
	.long	.Linfo_string48                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	153                             # DW_AT_decl_line
	.byte	23                              # Abbrev [23] 0x6a5:0x5 DW_TAG_pointer_type
	.long	1706                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x6aa:0x5 DW_TAG_structure_type
	.long	.Linfo_string50                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	23                              # Abbrev [23] 0x6af:0x5 DW_TAG_pointer_type
	.long	1716                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x6b4:0x5 DW_TAG_structure_type
	.long	.Linfo_string52                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	3                               # Abbrev [3] 0x6b9:0xc DW_TAG_array_type
	.long	75                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x6be:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	20                              # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	29                              # Abbrev [29] 0x6c5:0x5 DW_TAG_restrict_type
	.long	1209                            # DW_AT_type
	.byte	19                              # Abbrev [19] 0x6ca:0x17 DW_TAG_subprogram
	.long	.Linfo_string60                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.short	258                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1248                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	20                              # Abbrev [20] 0x6d6:0x5 DW_TAG_formal_parameter
	.long	1733                            # DW_AT_type
	.byte	20                              # Abbrev [20] 0x6db:0x5 DW_TAG_formal_parameter
	.long	1733                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	32                              # Abbrev [32] 0x6e1:0x11 DW_TAG_subprogram
	.long	.Linfo_string61                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	178                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1099                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	20                              # Abbrev [20] 0x6ec:0x5 DW_TAG_formal_parameter
	.long	1248                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	26                              # Abbrev [26] 0x6f2:0xe DW_TAG_subprogram
	.long	.Linfo_string62                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	568                             # DW_AT_decl_line
                                        # DW_AT_prototyped
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	20                              # Abbrev [20] 0x6fa:0x5 DW_TAG_formal_parameter
	.long	1057                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x700:0x5 DW_TAG_pointer_type
	.long	1621                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x705:0x5 DW_TAG_restrict_type
	.long	1802                            # DW_AT_type
	.byte	23                              # Abbrev [23] 0x70a:0x5 DW_TAG_pointer_type
	.long	1807                            # DW_AT_type
	.byte	3                               # Abbrev [3] 0x70f:0xd DW_TAG_array_type
	.long	1820                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x714:0x7 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.short	5000                            # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x71c:0x7 DW_TAG_base_type
	.long	.Linfo_string69                 # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang based Intel(R) oneAPI DPC++/C++ Compiler 2023.2.0 (2023.2.0.20230721)" # string offset=0
.Linfo_string1:
	.asciz	" --intel -g -O3 -x Host -S -D ENABLE_TIMING src/matmul_seq_ikj.c -fveclib=SVML -fheinous-gnu-extensions" # string offset=76
.Linfo_string2:
	.asciz	"src/matmul_seq_ikj.c"          # string offset=180
.Linfo_string3:
	.asciz	"/home/Zsf/Desktop/_FINAL_PROJECT" # string offset=201
.Linfo_string4:
	.asciz	"char"                          # string offset=234
.Linfo_string5:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=239
.Linfo_string6:
	.asciz	"aligned_alloc"                 # string offset=259
.Linfo_string7:
	.asciz	"unsigned long"                 # string offset=273
.Linfo_string8:
	.asciz	"size_t"                        # string offset=287
.Linfo_string9:
	.asciz	"clock_gettime"                 # string offset=294
.Linfo_string10:
	.asciz	"int"                           # string offset=308
.Linfo_string11:
	.asciz	"__clockid_t"                   # string offset=312
.Linfo_string12:
	.asciz	"clockid_t"                     # string offset=324
.Linfo_string13:
	.asciz	"tv_sec"                        # string offset=334
.Linfo_string14:
	.asciz	"long"                          # string offset=341
.Linfo_string15:
	.asciz	"__time_t"                      # string offset=346
.Linfo_string16:
	.asciz	"tv_nsec"                       # string offset=355
.Linfo_string17:
	.asciz	"__syscall_slong_t"             # string offset=363
.Linfo_string18:
	.asciz	"timespec"                      # string offset=381
.Linfo_string19:
	.asciz	"perror"                        # string offset=390
.Linfo_string20:
	.asciz	"fprintf"                       # string offset=397
.Linfo_string21:
	.asciz	"_flags"                        # string offset=405
.Linfo_string22:
	.asciz	"_IO_read_ptr"                  # string offset=412
.Linfo_string23:
	.asciz	"_IO_read_end"                  # string offset=425
.Linfo_string24:
	.asciz	"_IO_read_base"                 # string offset=438
.Linfo_string25:
	.asciz	"_IO_write_base"                # string offset=452
.Linfo_string26:
	.asciz	"_IO_write_ptr"                 # string offset=467
.Linfo_string27:
	.asciz	"_IO_write_end"                 # string offset=481
.Linfo_string28:
	.asciz	"_IO_buf_base"                  # string offset=495
.Linfo_string29:
	.asciz	"_IO_buf_end"                   # string offset=508
.Linfo_string30:
	.asciz	"_IO_save_base"                 # string offset=520
.Linfo_string31:
	.asciz	"_IO_backup_base"               # string offset=534
.Linfo_string32:
	.asciz	"_IO_save_end"                  # string offset=550
.Linfo_string33:
	.asciz	"_markers"                      # string offset=563
.Linfo_string34:
	.asciz	"_IO_marker"                    # string offset=572
.Linfo_string35:
	.asciz	"_chain"                        # string offset=583
.Linfo_string36:
	.asciz	"_fileno"                       # string offset=590
.Linfo_string37:
	.asciz	"_flags2"                       # string offset=598
.Linfo_string38:
	.asciz	"_old_offset"                   # string offset=606
.Linfo_string39:
	.asciz	"__off_t"                       # string offset=618
.Linfo_string40:
	.asciz	"_cur_column"                   # string offset=626
.Linfo_string41:
	.asciz	"unsigned short"                # string offset=638
.Linfo_string42:
	.asciz	"_vtable_offset"                # string offset=653
.Linfo_string43:
	.asciz	"signed char"                   # string offset=668
.Linfo_string44:
	.asciz	"_shortbuf"                     # string offset=680
.Linfo_string45:
	.asciz	"_lock"                         # string offset=690
.Linfo_string46:
	.asciz	"_IO_lock_t"                    # string offset=696
.Linfo_string47:
	.asciz	"_offset"                       # string offset=707
.Linfo_string48:
	.asciz	"__off64_t"                     # string offset=715
.Linfo_string49:
	.asciz	"_codecvt"                      # string offset=725
.Linfo_string50:
	.asciz	"_IO_codecvt"                   # string offset=734
.Linfo_string51:
	.asciz	"_wide_data"                    # string offset=746
.Linfo_string52:
	.asciz	"_IO_wide_data"                 # string offset=757
.Linfo_string53:
	.asciz	"_freeres_list"                 # string offset=771
.Linfo_string54:
	.asciz	"_freeres_buf"                  # string offset=785
.Linfo_string55:
	.asciz	"__pad5"                        # string offset=798
.Linfo_string56:
	.asciz	"_mode"                         # string offset=805
.Linfo_string57:
	.asciz	"_unused2"                      # string offset=811
.Linfo_string58:
	.asciz	"_IO_FILE"                      # string offset=820
.Linfo_string59:
	.asciz	"FILE"                          # string offset=829
.Linfo_string60:
	.asciz	"fopen"                         # string offset=834
.Linfo_string61:
	.asciz	"fclose"                        # string offset=840
.Linfo_string62:
	.asciz	"free"                          # string offset=847
.Linfo_string63:
	.asciz	"main"                          # string offset=852
.Linfo_string64:
	.asciz	"start"                         # string offset=857
.Linfo_string65:
	.asciz	"end"                           # string offset=863
.Linfo_string66:
	.asciz	"argc"                          # string offset=867
.Linfo_string67:
	.asciz	"argv"                          # string offset=872
.Linfo_string68:
	.asciz	"a"                             # string offset=877
.Linfo_string69:
	.asciz	"double"                        # string offset=879
.Linfo_string70:
	.asciz	"b"                             # string offset=886
.Linfo_string71:
	.asciz	"c"                             # string offset=888
.Linfo_string72:
	.asciz	"i"                             # string offset=890
.Linfo_string73:
	.asciz	"time_taken"                    # string offset=892
.Linfo_string74:
	.asciz	"f"                             # string offset=903
.Linfo_string75:
	.asciz	"j"                             # string offset=905
.Linfo_string76:
	.asciz	"k"                             # string offset=907
	.ident	"Intel(R) oneAPI DPC++/C++ Compiler 2023.2.0 (2023.2.0.20230721)"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
