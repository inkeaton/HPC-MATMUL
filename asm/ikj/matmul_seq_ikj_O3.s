	.text
	.file	"matmul_seq_ikj.c"
	.file	1 "/home/Zsf/Desktop/_FINAL_PROJECT" "src/matmul_seq_ikj.c"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function main
.LCPI0_0:
	.quad	0x4000000000000000              #  2
	.quad	0x4000000000000000              #  2
.LCPI0_1:
	.quad	0x4008000000000000              #  3
	.quad	0x4008000000000000              #  3
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
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
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	stmxcsr	(%rsp)
	orl	$32832, (%rsp)                  # imm = 0x8040
	ldmxcsr	(%rsp)
.Ltmp0:
	.loc	1 47 32 prologue_end            # src/matmul_seq_ikj.c:47:32
	movl	$64, %edi
.Ltmp1:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	movl	$200000000, %esi                # imm = 0xBEBC200
.Ltmp2:
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	callq	aligned_alloc
.Ltmp3:
	movq	%rax, %rbx
.Ltmp4:
	#DEBUG_VALUE: main:a <- $rbx
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
	testq	%rbx, %rbx
	.loc	1 51 12 is_stmt 0               # src/matmul_seq_ikj.c:51:12
	je	.LBB0_13
.Ltmp8:
# %bb.1:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $rax
	testq	%r14, %r14
	je	.LBB0_13
.Ltmp9:
# %bb.2:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $rax
	.loc	1 49 32 is_stmt 1               # src/matmul_seq_ikj.c:49:32
	movq	%rax, %r15
.Ltmp10:
	.loc	1 51 12                         # src/matmul_seq_ikj.c:51:12
	testq	%rax, %rax
	je	.LBB0_13
.Ltmp11:
# %bb.3:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r15
	.loc	1 63 21                         # src/matmul_seq_ikj.c:63:21
	movl	$200000000, %edx                # imm = 0xBEBC200
	movq	%r15, %rdi
	xorl	%esi, %esi
	callq	_intel_fast_memset@PLT
.Ltmp12:
	.loc	1 0 21 is_stmt 0                # src/matmul_seq_ikj.c:0:21
	movl	$48, %eax
	.loc	1 61 21 is_stmt 1 discriminator 2 # src/matmul_seq_ikj.c:61:21
	movapd	.LCPI0_0(%rip), %xmm0           # xmm0 = [2.0E+0,2.0E+0]
	.loc	1 62 21 discriminator 2         # src/matmul_seq_ikj.c:62:21
	movapd	.LCPI0_1(%rip), %xmm1           # xmm1 = [3.0E+0,3.0E+0]
.Ltmp13:
	.p2align	4, 0x90
.LBB0_4:                                # =>This Inner Loop Header: Depth=1
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r15
	.loc	1 61 21                         # src/matmul_seq_ikj.c:61:21
	movapd	%xmm0, -48(%rbx,%rax)
	.loc	1 62 21                         # src/matmul_seq_ikj.c:62:21
	movapd	%xmm1, -48(%r14,%rax)
	.loc	1 61 21                         # src/matmul_seq_ikj.c:61:21
	movapd	%xmm0, -32(%rbx,%rax)
	.loc	1 62 21                         # src/matmul_seq_ikj.c:62:21
	movapd	%xmm1, -32(%r14,%rax)
	.loc	1 61 21                         # src/matmul_seq_ikj.c:61:21
	movapd	%xmm0, -16(%rbx,%rax)
	.loc	1 62 21                         # src/matmul_seq_ikj.c:62:21
	movapd	%xmm1, -16(%r14,%rax)
	.loc	1 61 21                         # src/matmul_seq_ikj.c:61:21
	movapd	%xmm0, (%rbx,%rax)
	.loc	1 62 21                         # src/matmul_seq_ikj.c:62:21
	movapd	%xmm1, (%r14,%rax)
.Ltmp14:
	.loc	1 59 27                         # src/matmul_seq_ikj.c:59:27
	addq	$64, %rax
	cmpq	$200000048, %rax                # imm = 0xBEBC230
.Ltmp15:
	.loc	1 59 9 is_stmt 0                # src/matmul_seq_ikj.c:59:9
	jne	.LBB0_4
.Ltmp16:
# %bb.5:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r15
	.loc	1 0 9                           # src/matmul_seq_ikj.c:0:9
	movq	%rsp, %rsi
	.loc	1 69 9 is_stmt 1                # src/matmul_seq_ikj.c:69:9
	movl	$1, %edi
	callq	clock_gettime
.Ltmp17:
	#DEBUG_VALUE: i <- 0
	.loc	1 0 9 is_stmt 0                 # src/matmul_seq_ikj.c:0:9
	movq	%r14, %rax
	addq	$16, %rax
	leaq	16(%r15), %rcx
	xorl	%edx, %edx
.Ltmp18:
	.p2align	4, 0x90
.LBB0_6:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_7 Depth 2
                                        #       Child Loop BB0_8 Depth 3
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r15
	#DEBUG_VALUE: i <- 0
	movq	%rax, %rsi
	xorl	%edi, %edi
.Ltmp19:
	.p2align	4, 0x90
.LBB0_7:                                #   Parent Loop BB0_6 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_8 Depth 3
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r15
	#DEBUG_VALUE: i <- 0
	.loc	1 76 28 is_stmt 1               # src/matmul_seq_ikj.c:76:28
	imulq	$40000, %rdx, %r8               # imm = 0x9C40
	addq	%rbx, %r8
	.loc	1 76 36 is_stmt 0               # src/matmul_seq_ikj.c:76:36
	movsd	(%r8,%rdi,8), %xmm0             # xmm0 = mem[0],zero
	unpcklpd	%xmm0, %xmm0                    # xmm0 = xmm0[0,0]
	movq	$-2, %r8
.Ltmp20:
	.p2align	4, 0x90
.LBB0_8:                                #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r15
	#DEBUG_VALUE: i <- 0
	.loc	1 0 36                          # src/matmul_seq_ikj.c:0:36
	movapd	(%rsi,%r8,8), %xmm1
	.loc	1 76 36                         # src/matmul_seq_ikj.c:76:36
	mulpd	%xmm0, %xmm1
	.loc	1 76 25                         # src/matmul_seq_ikj.c:76:25
	addpd	(%rcx,%r8,8), %xmm1
	movapd	%xmm1, (%rcx,%r8,8)
	movapd	16(%rsi,%r8,8), %xmm1
	.loc	1 76 36                         # src/matmul_seq_ikj.c:76:36
	mulpd	%xmm0, %xmm1
	.loc	1 76 25                         # src/matmul_seq_ikj.c:76:25
	addpd	16(%rcx,%r8,8), %xmm1
	movapd	%xmm1, 16(%rcx,%r8,8)
	.loc	1 75 31 is_stmt 1               # src/matmul_seq_ikj.c:75:31
	addq	$4, %r8
	cmpq	$4998, %r8                      # imm = 0x1386
.Ltmp21:
	.loc	1 75 13 is_stmt 0               # src/matmul_seq_ikj.c:75:13
	jb	.LBB0_8
.Ltmp22:
# %bb.9:                                #   in Loop: Header=BB0_7 Depth=2
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r15
	#DEBUG_VALUE: i <- 0
	.loc	1 74 27 is_stmt 1               # src/matmul_seq_ikj.c:74:27
	leaq	1(%rdi), %r8
.Ltmp23:
	.loc	1 74 9 is_stmt 0                # src/matmul_seq_ikj.c:74:9
	addq	$40000, %rsi                    # imm = 0x9C40
.Ltmp24:
	.loc	1 74 27                         # src/matmul_seq_ikj.c:74:27
	cmpq	$4999, %rdi                     # imm = 0x1387
	movq	%r8, %rdi
.Ltmp25:
	.loc	1 74 9                          # src/matmul_seq_ikj.c:74:9
	jne	.LBB0_7
.Ltmp26:
# %bb.10:                               #   in Loop: Header=BB0_6 Depth=1
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r15
	#DEBUG_VALUE: i <- 0
	.loc	1 73 23 is_stmt 1               # src/matmul_seq_ikj.c:73:23
	leaq	1(%rdx), %rsi
.Ltmp27:
	.loc	1 73 5 is_stmt 0                # src/matmul_seq_ikj.c:73:5
	addq	$40000, %rcx                    # imm = 0x9C40
.Ltmp28:
	.loc	1 73 23                         # src/matmul_seq_ikj.c:73:23
	cmpq	$4999, %rdx                     # imm = 0x1387
	movq	%rsi, %rdx
.Ltmp29:
	.loc	1 73 5                          # src/matmul_seq_ikj.c:73:5
	jne	.LBB0_6
.Ltmp30:
# %bb.11:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r15
	.loc	1 0 5                           # src/matmul_seq_ikj.c:0:5
	movl	$1, %ebp
	leaq	24(%rsp), %rsi
	.loc	1 81 9 is_stmt 1                # src/matmul_seq_ikj.c:81:9
	movl	$1, %edi
	callq	clock_gettime
.Ltmp31:
	.loc	1 82 34                         # src/matmul_seq_ikj.c:82:34
	movq	24(%rsp), %rax
	.loc	1 82 64 is_stmt 0               # src/matmul_seq_ikj.c:82:64
	movq	32(%rsp), %rcx
	.loc	1 82 41                         # src/matmul_seq_ikj.c:82:41
	subq	(%rsp), %rax
	.loc	1 82 29                         # src/matmul_seq_ikj.c:82:29
	xorps	%xmm1, %xmm1
	cvtsi2sd	%rax, %xmm1
	.loc	1 82 72                         # src/matmul_seq_ikj.c:82:72
	subq	8(%rsp), %rcx
	.loc	1 82 59                         # src/matmul_seq_ikj.c:82:59
	xorps	%xmm0, %xmm0
	cvtsi2sd	%rcx, %xmm0
	.loc	1 82 89                         # src/matmul_seq_ikj.c:82:89
	mulsd	.LCPI0_2(%rip), %xmm0
	.loc	1 82 57                         # src/matmul_seq_ikj.c:82:57
	addsd	%xmm1, %xmm0
.Ltmp32:
	#DEBUG_VALUE: main:time_taken <- $xmm0
	.loc	1 84 17 is_stmt 1               # src/matmul_seq_ikj.c:84:17
	movq	stderr(%rip), %rdi
	.loc	1 84 9 is_stmt 0                # src/matmul_seq_ikj.c:84:9
	movl	$.L.str.1, %esi
	movl	$5000, %edx                     # imm = 0x1388
	movb	$1, %al
	callq	fprintf
.Ltmp33:
	.loc	1 89 15 is_stmt 1               # src/matmul_seq_ikj.c:89:15
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %esi
	callq	fopen
.Ltmp34:
	#DEBUG_VALUE: main:f <- $rax
	.loc	1 90 10                         # src/matmul_seq_ikj.c:90:10
	testq	%rax, %rax
.Ltmp35:
	.loc	1 90 9 is_stmt 0                # src/matmul_seq_ikj.c:90:9
	je	.LBB0_12
.Ltmp36:
# %bb.15:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r15
	#DEBUG_VALUE: main:f <- $rax
	.loc	1 89 15 is_stmt 1               # src/matmul_seq_ikj.c:89:15
	movq	%rax, %r12
	xorl	%r13d, %r13d
	.loc	1 96 5                          # src/matmul_seq_ikj.c:96:5
	movl	$.L.str.5, %esi
	movq	%rax, %rdi
	movl	$5000, %edx                     # imm = 0x1388
	xorl	%eax, %eax
.Ltmp37:
	#DEBUG_VALUE: main:f <- $r12
	callq	fprintf
.Ltmp38:
	#DEBUG_VALUE: i <- 0
	.loc	1 0 5 is_stmt 0                 # src/matmul_seq_ikj.c:0:5
	movq	%r15, 16(%rsp)                  # 8-byte Spill
.Ltmp39:
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 16, DW_OP_deref] $rsp
	movq	%r15, %rbp
	addq	$56, %rbp
.Ltmp40:
	.p2align	4, 0x90
.LBB0_16:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_17 Depth 2
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 16, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	#DEBUG_VALUE: i <- 0
	xorl	%r15d, %r15d
.Ltmp41:
	.p2align	4, 0x90
.LBB0_17:                               #   Parent Loop BB0_16 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 16, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	#DEBUG_VALUE: i <- 0
	.loc	1 101 33 is_stmt 1              # src/matmul_seq_ikj.c:101:33
	movsd	-56(%rbp,%r15), %xmm0           # xmm0 = mem[0],zero
	.loc	1 101 13 is_stmt 0              # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp42:
	.loc	1 101 33                        # src/matmul_seq_ikj.c:101:33
	movsd	-48(%rbp,%r15), %xmm0           # xmm0 = mem[0],zero
	.loc	1 101 13                        # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp43:
	.loc	1 101 33                        # src/matmul_seq_ikj.c:101:33
	movsd	-40(%rbp,%r15), %xmm0           # xmm0 = mem[0],zero
	.loc	1 101 13                        # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp44:
	.loc	1 101 33                        # src/matmul_seq_ikj.c:101:33
	movsd	-32(%rbp,%r15), %xmm0           # xmm0 = mem[0],zero
	.loc	1 101 13                        # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp45:
	.loc	1 101 33                        # src/matmul_seq_ikj.c:101:33
	movsd	-24(%rbp,%r15), %xmm0           # xmm0 = mem[0],zero
	.loc	1 101 13                        # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp46:
	.loc	1 101 33                        # src/matmul_seq_ikj.c:101:33
	movsd	-16(%rbp,%r15), %xmm0           # xmm0 = mem[0],zero
	.loc	1 101 13                        # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp47:
	.loc	1 101 33                        # src/matmul_seq_ikj.c:101:33
	movsd	-8(%rbp,%r15), %xmm0            # xmm0 = mem[0],zero
	.loc	1 101 13                        # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp48:
	.loc	1 101 33                        # src/matmul_seq_ikj.c:101:33
	movsd	(%rbp,%r15), %xmm0              # xmm0 = mem[0],zero
	.loc	1 101 13                        # src/matmul_seq_ikj.c:101:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp49:
	.loc	1 99 27 is_stmt 1               # src/matmul_seq_ikj.c:99:27
	addq	$64, %r15
	cmpq	$8000, %r15                     # imm = 0x1F40
.Ltmp50:
	.loc	1 99 9 is_stmt 0                # src/matmul_seq_ikj.c:99:9
	jne	.LBB0_17
.Ltmp51:
# %bb.18:                               #   in Loop: Header=BB0_16 Depth=1
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 16, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	#DEBUG_VALUE: i <- 0
	.loc	1 103 9 is_stmt 1               # src/matmul_seq_ikj.c:103:9
	movl	$10, %edi
	movq	%r12, %rsi
	callq	fputc@PLT
.Ltmp52:
	.loc	1 97 23                         # src/matmul_seq_ikj.c:97:23
	leaq	1(%r13), %rax
.Ltmp53:
	.loc	1 97 5 is_stmt 0                # src/matmul_seq_ikj.c:97:5
	addq	$40000, %rbp                    # imm = 0x9C40
.Ltmp54:
	.loc	1 97 23                         # src/matmul_seq_ikj.c:97:23
	cmpq	$999, %r13                      # imm = 0x3E7
	movq	%rax, %r13
.Ltmp55:
	.loc	1 97 5                          # src/matmul_seq_ikj.c:97:5
	jne	.LBB0_16
.Ltmp56:
# %bb.19:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 16, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	.loc	1 106 5 is_stmt 1               # src/matmul_seq_ikj.c:106:5
	movq	%r12, %rdi
	callq	fclose
.Ltmp57:
	.loc	1 109 5                         # src/matmul_seq_ikj.c:109:5
	movq	%rbx, %rdi
	callq	free
.Ltmp58:
	.loc	1 110 5                         # src/matmul_seq_ikj.c:110:5
	movq	%r14, %rdi
	callq	free
.Ltmp59:
	.loc	1 0 5 is_stmt 0                 # src/matmul_seq_ikj.c:0:5
	movq	16(%rsp), %rdi                  # 8-byte Reload
	.loc	1 111 5 is_stmt 1               # src/matmul_seq_ikj.c:111:5
	callq	free
.Ltmp60:
	.loc	1 0 5 is_stmt 0                 # src/matmul_seq_ikj.c:0:5
	xorl	%ebp, %ebp
	jmp	.LBB0_14
.Ltmp61:
.LBB0_13:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $rax
	.loc	1 52 9 is_stmt 1                # src/matmul_seq_ikj.c:52:9
	movl	$.L.str, %edi
	callq	perror
.Ltmp62:
	.loc	1 0 9 is_stmt 0                 # src/matmul_seq_ikj.c:0:9
	movl	$1, %ebp
.Ltmp63:
.LBB0_14:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	.loc	1 113 1 is_stmt 1               # src/matmul_seq_ikj.c:113:1
	movl	%ebp, %eax
	.loc	1 113 1 epilogue_begin is_stmt 0 # src/matmul_seq_ikj.c:113:1
	addq	$40, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
.Ltmp64:
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
.Ltmp65:
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Ltmp66:
.LBB0_12:
	.cfi_def_cfa_offset 96
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r15
	#DEBUG_VALUE: main:f <- $rax
	.loc	1 92 9 is_stmt 1                # src/matmul_seq_ikj.c:92:9
	movl	$.L.str.4, %edi
	callq	perror
.Ltmp67:
	.loc	1 0 9 is_stmt 0                 # src/matmul_seq_ikj.c:0:9
	jmp	.LBB0_14
.Ltmp68:
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
	.quad	.Ltmp2-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	84                              # DW_OP_reg4
	.quad	.Ltmp2-.Lfunc_begin0
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
	.quad	.Ltmp64-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	83                              # DW_OP_reg3
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	83                              # DW_OP_reg3
	.quad	0
	.quad	0
.Ldebug_loc3:
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	94                              # DW_OP_reg14
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	94                              # DW_OP_reg14
	.quad	0
	.quad	0
.Ldebug_loc4:
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	80                              # DW_OP_reg0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	95                              # DW_OP_reg15
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.short	2                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	16                              # 16
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	80                              # DW_OP_reg0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	95                              # DW_OP_reg15
	.quad	0
	.quad	0
.Ldebug_loc5:
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	97                              # DW_OP_reg17
	.quad	0
	.quad	0
.Ldebug_loc6:
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	80                              # DW_OP_reg0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	92                              # DW_OP_reg12
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
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
	.byte	15                              # Abbreviation Code
	.ascii	"\211\202\001"                  # DW_TAG_GNU_call_site
	.byte	1                               # DW_CHILDREN_yes
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	16                              # Abbreviation Code
	.ascii	"\212\202\001"                  # DW_TAG_GNU_call_site_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\221B"                         # DW_AT_GNU_call_site_value
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	17                              # Abbreviation Code
	.ascii	"\211\202\001"                  # DW_TAG_GNU_call_site
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	18                              # Abbreviation Code
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
	.byte	19                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	20                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	21                              # Abbreviation Code
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
	.byte	22                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	23                              # Abbreviation Code
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
	.byte	24                              # Abbreviation Code
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
	.byte	25                              # Abbreviation Code
	.byte	24                              # DW_TAG_unspecified_parameters
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	26                              # Abbreviation Code
	.byte	55                              # DW_TAG_restrict_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	27                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	28                              # Abbreviation Code
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
	.byte	29                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	30                              # Abbreviation Code
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
	.byte	31                              # Abbreviation Code
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
	.byte	32                              # Abbreviation Code
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
	.byte	1                               # Abbrev [1] 0xb:0x72d DW_TAG_compile_unit
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
	.byte	8                               # Abbrev [8] 0x102:0x31c DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string63                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	38                              # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1119                            # DW_AT_type
                                        # DW_AT_external
	.byte	9                               # Abbrev [9] 0x11b:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc0                    # DW_AT_location
	.long	.Linfo_string66                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	38                              # DW_AT_decl_line
	.long	1119                            # DW_AT_type
	.byte	9                               # Abbrev [9] 0x12a:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc1                    # DW_AT_location
	.long	.Linfo_string67                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	38                              # DW_AT_decl_line
	.long	1812                            # DW_AT_type
	.byte	10                              # Abbrev [10] 0x139:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	0
	.long	.Linfo_string64                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	68                              # DW_AT_decl_line
	.long	1153                            # DW_AT_type
	.byte	10                              # Abbrev [10] 0x147:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	24
	.long	.Linfo_string65                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	68                              # DW_AT_decl_line
	.long	1153                            # DW_AT_type
	.byte	11                              # Abbrev [11] 0x155:0xf DW_TAG_variable
	.long	.Ldebug_loc2                    # DW_AT_location
	.long	.Linfo_string68                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	47                              # DW_AT_decl_line
	.long	1817                            # DW_AT_type
	.byte	11                              # Abbrev [11] 0x164:0xf DW_TAG_variable
	.long	.Ldebug_loc3                    # DW_AT_location
	.long	.Linfo_string70                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	48                              # DW_AT_decl_line
	.long	1817                            # DW_AT_type
	.byte	11                              # Abbrev [11] 0x173:0xf DW_TAG_variable
	.long	.Ldebug_loc4                    # DW_AT_location
	.long	.Linfo_string71                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
	.long	1817                            # DW_AT_type
	.byte	11                              # Abbrev [11] 0x182:0xf DW_TAG_variable
	.long	.Ldebug_loc5                    # DW_AT_location
	.long	.Linfo_string73                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	82                              # DW_AT_decl_line
	.long	1840                            # DW_AT_type
	.byte	11                              # Abbrev [11] 0x191:0xf DW_TAG_variable
	.long	.Ldebug_loc6                    # DW_AT_location
	.long	.Linfo_string74                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	89                              # DW_AT_decl_line
	.long	1244                            # DW_AT_type
	.byte	12                              # Abbrev [12] 0x1a0:0x32 DW_TAG_lexical_block
	.quad	.Ltmp11                         # DW_AT_low_pc
	.long	.Ltmp16-.Ltmp11                 # DW_AT_high_pc
	.byte	13                              # Abbrev [13] 0x1ad:0xb DW_TAG_variable
	.long	.Linfo_string72                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.long	1119                            # DW_AT_type
	.byte	12                              # Abbrev [12] 0x1b8:0x19 DW_TAG_lexical_block
	.quad	.Ltmp11                         # DW_AT_low_pc
	.long	.Ltmp16-.Ltmp11                 # DW_AT_high_pc
	.byte	13                              # Abbrev [13] 0x1c5:0xb DW_TAG_variable
	.long	.Linfo_string75                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
	.long	1119                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1d2:0x4c DW_TAG_lexical_block
	.quad	.Ltmp19                         # DW_AT_low_pc
	.long	.Ltmp30-.Ltmp19                 # DW_AT_high_pc
	.byte	14                              # Abbrev [14] 0x1df:0xc DW_TAG_variable
	.byte	0                               # DW_AT_const_value
	.long	.Linfo_string72                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.long	1119                            # DW_AT_type
	.byte	12                              # Abbrev [12] 0x1eb:0x32 DW_TAG_lexical_block
	.quad	.Ltmp19                         # DW_AT_low_pc
	.long	.Ltmp26-.Ltmp19                 # DW_AT_high_pc
	.byte	13                              # Abbrev [13] 0x1f8:0xb DW_TAG_variable
	.long	.Linfo_string76                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	74                              # DW_AT_decl_line
	.long	1119                            # DW_AT_type
	.byte	12                              # Abbrev [12] 0x203:0x19 DW_TAG_lexical_block
	.quad	.Ltmp19                         # DW_AT_low_pc
	.long	.Ltmp22-.Ltmp19                 # DW_AT_high_pc
	.byte	13                              # Abbrev [13] 0x210:0xb DW_TAG_variable
	.long	.Linfo_string75                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
	.long	1119                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x21e:0x33 DW_TAG_lexical_block
	.quad	.Ltmp41                         # DW_AT_low_pc
	.long	.Ltmp56-.Ltmp41                 # DW_AT_high_pc
	.byte	14                              # Abbrev [14] 0x22b:0xc DW_TAG_variable
	.byte	0                               # DW_AT_const_value
	.long	.Linfo_string72                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	97                              # DW_AT_decl_line
	.long	1119                            # DW_AT_type
	.byte	12                              # Abbrev [12] 0x237:0x19 DW_TAG_lexical_block
	.quad	.Ltmp41                         # DW_AT_low_pc
	.long	.Ltmp51-.Ltmp41                 # DW_AT_high_pc
	.byte	13                              # Abbrev [13] 0x244:0xb DW_TAG_variable
	.long	.Linfo_string75                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	99                              # DW_AT_decl_line
	.long	1119                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x251:0x1d DW_TAG_GNU_call_site
	.long	1054                            # DW_AT_abstract_origin
	.quad	.Ltmp3                          # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x25e:0x9 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	5                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\200\204\257_"
	.byte	16                              # Abbrev [16] 0x267:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	16
	.byte	64
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x26e:0x1d DW_TAG_GNU_call_site
	.long	1054                            # DW_AT_abstract_origin
	.quad	.Ltmp5                          # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x27b:0x9 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	5                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\200\204\257_"
	.byte	16                              # Abbrev [16] 0x284:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	16
	.byte	64
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x28b:0x1d DW_TAG_GNU_call_site
	.long	1054                            # DW_AT_abstract_origin
	.quad	.Ltmp7                          # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x298:0x9 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	5                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\200\204\257_"
	.byte	16                              # Abbrev [16] 0x2a1:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	16
	.byte	64
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x2a8:0x19 DW_TAG_GNU_call_site
	.long	1096                            # DW_AT_abstract_origin
	.quad	.Ltmp17                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x2b5:0x5 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	1                               # DW_AT_GNU_call_site_value
	.byte	49
	.byte	16                              # Abbrev [16] 0x2ba:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	145
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x2c1:0x19 DW_TAG_GNU_call_site
	.long	1096                            # DW_AT_abstract_origin
	.quad	.Ltmp31                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x2ce:0x5 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	1                               # DW_AT_GNU_call_site_value
	.byte	49
	.byte	16                              # Abbrev [16] 0x2d3:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	145
	.byte	24
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x2da:0x15 DW_TAG_GNU_call_site
	.long	1215                            # DW_AT_abstract_origin
	.quad	.Ltmp33                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x2e7:0x7 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	81
	.byte	3                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\210'"
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0x2ef:0xd DW_TAG_GNU_call_site
	.long	1744                            # DW_AT_abstract_origin
	.quad	.Ltmp34                         # DW_AT_low_pc
	.byte	15                              # Abbrev [15] 0x2fc:0x15 DW_TAG_GNU_call_site
	.long	1215                            # DW_AT_abstract_origin
	.quad	.Ltmp38                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x309:0x7 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	81
	.byte	3                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\210'"
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x311:0x14 DW_TAG_GNU_call_site
	.long	1215                            # DW_AT_abstract_origin
	.quad	.Ltmp42                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x31e:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x325:0x14 DW_TAG_GNU_call_site
	.long	1215                            # DW_AT_abstract_origin
	.quad	.Ltmp43                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x332:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x339:0x14 DW_TAG_GNU_call_site
	.long	1215                            # DW_AT_abstract_origin
	.quad	.Ltmp44                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x346:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x34d:0x14 DW_TAG_GNU_call_site
	.long	1215                            # DW_AT_abstract_origin
	.quad	.Ltmp45                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x35a:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x361:0x14 DW_TAG_GNU_call_site
	.long	1215                            # DW_AT_abstract_origin
	.quad	.Ltmp46                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x36e:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x375:0x14 DW_TAG_GNU_call_site
	.long	1215                            # DW_AT_abstract_origin
	.quad	.Ltmp47                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x382:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x389:0x14 DW_TAG_GNU_call_site
	.long	1215                            # DW_AT_abstract_origin
	.quad	.Ltmp48                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x396:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x39d:0x14 DW_TAG_GNU_call_site
	.long	1215                            # DW_AT_abstract_origin
	.quad	.Ltmp49                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x3aa:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x3b1:0x14 DW_TAG_GNU_call_site
	.long	1767                            # DW_AT_abstract_origin
	.quad	.Ltmp57                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x3be:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x3c5:0x14 DW_TAG_GNU_call_site
	.long	1784                            # DW_AT_abstract_origin
	.quad	.Ltmp58                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x3d2:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	115
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x3d9:0x14 DW_TAG_GNU_call_site
	.long	1784                            # DW_AT_abstract_origin
	.quad	.Ltmp59                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x3e6:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	126
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0x3ed:0x16 DW_TAG_GNU_call_site
	.long	1784                            # DW_AT_abstract_origin
	.quad	.Ltmp60                         # DW_AT_low_pc
	.byte	16                              # Abbrev [16] 0x3fa:0x8 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	4                               # DW_AT_GNU_call_site_value
	.byte	145
	.byte	16
	.byte	148
	.byte	8
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0x403:0xd DW_TAG_GNU_call_site
	.long	1798                            # DW_AT_abstract_origin
	.quad	.Ltmp62                         # DW_AT_low_pc
	.byte	17                              # Abbrev [17] 0x410:0xd DW_TAG_GNU_call_site
	.long	1798                            # DW_AT_abstract_origin
	.quad	.Ltmp67                         # DW_AT_low_pc
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0x41e:0x17 DW_TAG_subprogram
	.long	.Linfo_string6                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	605                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1077                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	19                              # Abbrev [19] 0x42a:0x5 DW_TAG_formal_parameter
	.long	1078                            # DW_AT_type
	.byte	19                              # Abbrev [19] 0x42f:0x5 DW_TAG_formal_parameter
	.long	1078                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	20                              # Abbrev [20] 0x435:0x1 DW_TAG_pointer_type
	.byte	21                              # Abbrev [21] 0x436:0xb DW_TAG_typedef
	.long	1089                            # DW_AT_type
	.long	.Linfo_string8                  # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
	.byte	5                               # Abbrev [5] 0x441:0x7 DW_TAG_base_type
	.long	.Linfo_string7                  # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	18                              # Abbrev [18] 0x448:0x17 DW_TAG_subprogram
	.long	.Linfo_string9                  # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.short	288                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1119                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	19                              # Abbrev [19] 0x454:0x5 DW_TAG_formal_parameter
	.long	1126                            # DW_AT_type
	.byte	19                              # Abbrev [19] 0x459:0x5 DW_TAG_formal_parameter
	.long	1148                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x45f:0x7 DW_TAG_base_type
	.long	.Linfo_string10                 # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	21                              # Abbrev [21] 0x466:0xb DW_TAG_typedef
	.long	1137                            # DW_AT_type
	.long	.Linfo_string12                 # DW_AT_name
	.byte	6                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x471:0xb DW_TAG_typedef
	.long	1119                            # DW_AT_type
	.long	.Linfo_string11                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	169                             # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x47c:0x5 DW_TAG_pointer_type
	.long	1153                            # DW_AT_type
	.byte	23                              # Abbrev [23] 0x481:0x21 DW_TAG_structure_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	16                              # DW_AT_byte_size
	.byte	7                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x489:0xc DW_TAG_member
	.long	.Linfo_string13                 # DW_AT_name
	.long	1186                            # DW_AT_type
	.byte	7                               # DW_AT_decl_file
	.byte	16                              # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x495:0xc DW_TAG_member
	.long	.Linfo_string16                 # DW_AT_name
	.long	1204                            # DW_AT_type
	.byte	7                               # DW_AT_decl_file
	.byte	21                              # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x4a2:0xb DW_TAG_typedef
	.long	1197                            # DW_AT_type
	.long	.Linfo_string15                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	160                             # DW_AT_decl_line
	.byte	5                               # Abbrev [5] 0x4ad:0x7 DW_TAG_base_type
	.long	.Linfo_string14                 # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	21                              # Abbrev [21] 0x4b4:0xb DW_TAG_typedef
	.long	1197                            # DW_AT_type
	.long	.Linfo_string17                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	197                             # DW_AT_decl_line
	.byte	18                              # Abbrev [18] 0x4bf:0x18 DW_TAG_subprogram
	.long	.Linfo_string19                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.short	350                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1119                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	19                              # Abbrev [19] 0x4cb:0x5 DW_TAG_formal_parameter
	.long	1239                            # DW_AT_type
	.byte	19                              # Abbrev [19] 0x4d0:0x5 DW_TAG_formal_parameter
	.long	1729                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x4d5:0x1 DW_TAG_unspecified_parameters
	.byte	0                               # End Of Children Mark
	.byte	26                              # Abbrev [26] 0x4d7:0x5 DW_TAG_restrict_type
	.long	1244                            # DW_AT_type
	.byte	22                              # Abbrev [22] 0x4dc:0x5 DW_TAG_pointer_type
	.long	1249                            # DW_AT_type
	.byte	21                              # Abbrev [21] 0x4e1:0xb DW_TAG_typedef
	.long	1260                            # DW_AT_type
	.long	.Linfo_string58                 # DW_AT_name
	.byte	10                              # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	23                              # Abbrev [23] 0x4ec:0x165 DW_TAG_structure_type
	.long	.Linfo_string57                 # DW_AT_name
	.byte	216                             # DW_AT_byte_size
	.byte	9                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
	.byte	24                              # Abbrev [24] 0x4f4:0xc DW_TAG_member
	.long	.Linfo_string20                 # DW_AT_name
	.long	1119                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	51                              # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x500:0xc DW_TAG_member
	.long	.Linfo_string21                 # DW_AT_name
	.long	1617                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x50c:0xc DW_TAG_member
	.long	.Linfo_string22                 # DW_AT_name
	.long	1617                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	55                              # DW_AT_decl_line
	.byte	16                              # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x518:0xc DW_TAG_member
	.long	.Linfo_string23                 # DW_AT_name
	.long	1617                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	56                              # DW_AT_decl_line
	.byte	24                              # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x524:0xc DW_TAG_member
	.long	.Linfo_string24                 # DW_AT_name
	.long	1617                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	57                              # DW_AT_decl_line
	.byte	32                              # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x530:0xc DW_TAG_member
	.long	.Linfo_string25                 # DW_AT_name
	.long	1617                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.byte	40                              # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x53c:0xc DW_TAG_member
	.long	.Linfo_string26                 # DW_AT_name
	.long	1617                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
	.byte	48                              # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x548:0xc DW_TAG_member
	.long	.Linfo_string27                 # DW_AT_name
	.long	1617                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	60                              # DW_AT_decl_line
	.byte	56                              # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x554:0xc DW_TAG_member
	.long	.Linfo_string28                 # DW_AT_name
	.long	1617                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	61                              # DW_AT_decl_line
	.byte	64                              # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x560:0xc DW_TAG_member
	.long	.Linfo_string29                 # DW_AT_name
	.long	1617                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	64                              # DW_AT_decl_line
	.byte	72                              # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x56c:0xc DW_TAG_member
	.long	.Linfo_string30                 # DW_AT_name
	.long	1617                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	65                              # DW_AT_decl_line
	.byte	80                              # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x578:0xc DW_TAG_member
	.long	.Linfo_string31                 # DW_AT_name
	.long	1617                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	66                              # DW_AT_decl_line
	.byte	88                              # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x584:0xc DW_TAG_member
	.long	.Linfo_string32                 # DW_AT_name
	.long	1622                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	68                              # DW_AT_decl_line
	.byte	96                              # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x590:0xc DW_TAG_member
	.long	.Linfo_string34                 # DW_AT_name
	.long	1632                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	70                              # DW_AT_decl_line
	.byte	104                             # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x59c:0xc DW_TAG_member
	.long	.Linfo_string35                 # DW_AT_name
	.long	1119                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	72                              # DW_AT_decl_line
	.byte	112                             # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x5a8:0xc DW_TAG_member
	.long	.Linfo_string36                 # DW_AT_name
	.long	1119                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.byte	116                             # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x5b4:0xc DW_TAG_member
	.long	.Linfo_string37                 # DW_AT_name
	.long	1637                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	74                              # DW_AT_decl_line
	.byte	120                             # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x5c0:0xc DW_TAG_member
	.long	.Linfo_string39                 # DW_AT_name
	.long	1648                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	77                              # DW_AT_decl_line
	.byte	128                             # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x5cc:0xc DW_TAG_member
	.long	.Linfo_string41                 # DW_AT_name
	.long	1655                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	78                              # DW_AT_decl_line
	.byte	130                             # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x5d8:0xc DW_TAG_member
	.long	.Linfo_string43                 # DW_AT_name
	.long	1662                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	79                              # DW_AT_decl_line
	.byte	131                             # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x5e4:0xc DW_TAG_member
	.long	.Linfo_string44                 # DW_AT_name
	.long	1674                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	81                              # DW_AT_decl_line
	.byte	136                             # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x5f0:0xc DW_TAG_member
	.long	.Linfo_string46                 # DW_AT_name
	.long	1686                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	89                              # DW_AT_decl_line
	.byte	144                             # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x5fc:0xc DW_TAG_member
	.long	.Linfo_string48                 # DW_AT_name
	.long	1697                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	91                              # DW_AT_decl_line
	.byte	152                             # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x608:0xc DW_TAG_member
	.long	.Linfo_string50                 # DW_AT_name
	.long	1707                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	92                              # DW_AT_decl_line
	.byte	160                             # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x614:0xc DW_TAG_member
	.long	.Linfo_string52                 # DW_AT_name
	.long	1632                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	93                              # DW_AT_decl_line
	.byte	168                             # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x620:0xc DW_TAG_member
	.long	.Linfo_string53                 # DW_AT_name
	.long	1077                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	94                              # DW_AT_decl_line
	.byte	176                             # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x62c:0xc DW_TAG_member
	.long	.Linfo_string54                 # DW_AT_name
	.long	1078                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	95                              # DW_AT_decl_line
	.byte	184                             # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x638:0xc DW_TAG_member
	.long	.Linfo_string55                 # DW_AT_name
	.long	1119                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	96                              # DW_AT_decl_line
	.byte	192                             # DW_AT_data_member_location
	.byte	24                              # Abbrev [24] 0x644:0xc DW_TAG_member
	.long	.Linfo_string56                 # DW_AT_name
	.long	1717                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	98                              # DW_AT_decl_line
	.byte	196                             # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	22                              # Abbrev [22] 0x651:0x5 DW_TAG_pointer_type
	.long	75                              # DW_AT_type
	.byte	22                              # Abbrev [22] 0x656:0x5 DW_TAG_pointer_type
	.long	1627                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x65b:0x5 DW_TAG_structure_type
	.long	.Linfo_string33                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	22                              # Abbrev [22] 0x660:0x5 DW_TAG_pointer_type
	.long	1260                            # DW_AT_type
	.byte	21                              # Abbrev [21] 0x665:0xb DW_TAG_typedef
	.long	1197                            # DW_AT_type
	.long	.Linfo_string38                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	152                             # DW_AT_decl_line
	.byte	5                               # Abbrev [5] 0x670:0x7 DW_TAG_base_type
	.long	.Linfo_string40                 # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	2                               # DW_AT_byte_size
	.byte	5                               # Abbrev [5] 0x677:0x7 DW_TAG_base_type
	.long	.Linfo_string42                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x67e:0xc DW_TAG_array_type
	.long	75                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x683:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	22                              # Abbrev [22] 0x68a:0x5 DW_TAG_pointer_type
	.long	1679                            # DW_AT_type
	.byte	28                              # Abbrev [28] 0x68f:0x7 DW_TAG_typedef
	.long	.Linfo_string45                 # DW_AT_name
	.byte	9                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
	.byte	21                              # Abbrev [21] 0x696:0xb DW_TAG_typedef
	.long	1197                            # DW_AT_type
	.long	.Linfo_string47                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	153                             # DW_AT_decl_line
	.byte	22                              # Abbrev [22] 0x6a1:0x5 DW_TAG_pointer_type
	.long	1702                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x6a6:0x5 DW_TAG_structure_type
	.long	.Linfo_string49                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	22                              # Abbrev [22] 0x6ab:0x5 DW_TAG_pointer_type
	.long	1712                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x6b0:0x5 DW_TAG_structure_type
	.long	.Linfo_string51                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	3                               # Abbrev [3] 0x6b5:0xc DW_TAG_array_type
	.long	75                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x6ba:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	20                              # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	26                              # Abbrev [26] 0x6c1:0x5 DW_TAG_restrict_type
	.long	1734                            # DW_AT_type
	.byte	22                              # Abbrev [22] 0x6c6:0x5 DW_TAG_pointer_type
	.long	1739                            # DW_AT_type
	.byte	29                              # Abbrev [29] 0x6cb:0x5 DW_TAG_const_type
	.long	75                              # DW_AT_type
	.byte	18                              # Abbrev [18] 0x6d0:0x17 DW_TAG_subprogram
	.long	.Linfo_string59                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.short	258                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1244                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	19                              # Abbrev [19] 0x6dc:0x5 DW_TAG_formal_parameter
	.long	1729                            # DW_AT_type
	.byte	19                              # Abbrev [19] 0x6e1:0x5 DW_TAG_formal_parameter
	.long	1729                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	30                              # Abbrev [30] 0x6e7:0x11 DW_TAG_subprogram
	.long	.Linfo_string60                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	178                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1119                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	19                              # Abbrev [19] 0x6f2:0x5 DW_TAG_formal_parameter
	.long	1244                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	31                              # Abbrev [31] 0x6f8:0xe DW_TAG_subprogram
	.long	.Linfo_string61                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	568                             # DW_AT_decl_line
                                        # DW_AT_prototyped
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	19                              # Abbrev [19] 0x700:0x5 DW_TAG_formal_parameter
	.long	1077                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	31                              # Abbrev [31] 0x706:0xe DW_TAG_subprogram
	.long	.Linfo_string62                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.short	804                             # DW_AT_decl_line
                                        # DW_AT_prototyped
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	19                              # Abbrev [19] 0x70e:0x5 DW_TAG_formal_parameter
	.long	1734                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	22                              # Abbrev [22] 0x714:0x5 DW_TAG_pointer_type
	.long	1617                            # DW_AT_type
	.byte	26                              # Abbrev [26] 0x719:0x5 DW_TAG_restrict_type
	.long	1822                            # DW_AT_type
	.byte	22                              # Abbrev [22] 0x71e:0x5 DW_TAG_pointer_type
	.long	1827                            # DW_AT_type
	.byte	3                               # Abbrev [3] 0x723:0xd DW_TAG_array_type
	.long	1840                            # DW_AT_type
	.byte	32                              # Abbrev [32] 0x728:0x7 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.short	5000                            # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x730:0x7 DW_TAG_base_type
	.long	.Linfo_string69                 # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang based Intel(R) oneAPI DPC++/C++ Compiler 2023.2.0 (2023.2.0.20230721)" # string offset=0
.Linfo_string1:
	.asciz	" --intel -g -O3 -S -D ENABLE_TIMING src/matmul_seq_ikj.c -fveclib=SVML -fheinous-gnu-extensions" # string offset=76
.Linfo_string2:
	.asciz	"src/matmul_seq_ikj.c"          # string offset=172
.Linfo_string3:
	.asciz	"/home/Zsf/Desktop/_FINAL_PROJECT" # string offset=193
.Linfo_string4:
	.asciz	"char"                          # string offset=226
.Linfo_string5:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=231
.Linfo_string6:
	.asciz	"aligned_alloc"                 # string offset=251
.Linfo_string7:
	.asciz	"unsigned long"                 # string offset=265
.Linfo_string8:
	.asciz	"size_t"                        # string offset=279
.Linfo_string9:
	.asciz	"clock_gettime"                 # string offset=286
.Linfo_string10:
	.asciz	"int"                           # string offset=300
.Linfo_string11:
	.asciz	"__clockid_t"                   # string offset=304
.Linfo_string12:
	.asciz	"clockid_t"                     # string offset=316
.Linfo_string13:
	.asciz	"tv_sec"                        # string offset=326
.Linfo_string14:
	.asciz	"long"                          # string offset=333
.Linfo_string15:
	.asciz	"__time_t"                      # string offset=338
.Linfo_string16:
	.asciz	"tv_nsec"                       # string offset=347
.Linfo_string17:
	.asciz	"__syscall_slong_t"             # string offset=355
.Linfo_string18:
	.asciz	"timespec"                      # string offset=373
.Linfo_string19:
	.asciz	"fprintf"                       # string offset=382
.Linfo_string20:
	.asciz	"_flags"                        # string offset=390
.Linfo_string21:
	.asciz	"_IO_read_ptr"                  # string offset=397
.Linfo_string22:
	.asciz	"_IO_read_end"                  # string offset=410
.Linfo_string23:
	.asciz	"_IO_read_base"                 # string offset=423
.Linfo_string24:
	.asciz	"_IO_write_base"                # string offset=437
.Linfo_string25:
	.asciz	"_IO_write_ptr"                 # string offset=452
.Linfo_string26:
	.asciz	"_IO_write_end"                 # string offset=466
.Linfo_string27:
	.asciz	"_IO_buf_base"                  # string offset=480
.Linfo_string28:
	.asciz	"_IO_buf_end"                   # string offset=493
.Linfo_string29:
	.asciz	"_IO_save_base"                 # string offset=505
.Linfo_string30:
	.asciz	"_IO_backup_base"               # string offset=519
.Linfo_string31:
	.asciz	"_IO_save_end"                  # string offset=535
.Linfo_string32:
	.asciz	"_markers"                      # string offset=548
.Linfo_string33:
	.asciz	"_IO_marker"                    # string offset=557
.Linfo_string34:
	.asciz	"_chain"                        # string offset=568
.Linfo_string35:
	.asciz	"_fileno"                       # string offset=575
.Linfo_string36:
	.asciz	"_flags2"                       # string offset=583
.Linfo_string37:
	.asciz	"_old_offset"                   # string offset=591
.Linfo_string38:
	.asciz	"__off_t"                       # string offset=603
.Linfo_string39:
	.asciz	"_cur_column"                   # string offset=611
.Linfo_string40:
	.asciz	"unsigned short"                # string offset=623
.Linfo_string41:
	.asciz	"_vtable_offset"                # string offset=638
.Linfo_string42:
	.asciz	"signed char"                   # string offset=653
.Linfo_string43:
	.asciz	"_shortbuf"                     # string offset=665
.Linfo_string44:
	.asciz	"_lock"                         # string offset=675
.Linfo_string45:
	.asciz	"_IO_lock_t"                    # string offset=681
.Linfo_string46:
	.asciz	"_offset"                       # string offset=692
.Linfo_string47:
	.asciz	"__off64_t"                     # string offset=700
.Linfo_string48:
	.asciz	"_codecvt"                      # string offset=710
.Linfo_string49:
	.asciz	"_IO_codecvt"                   # string offset=719
.Linfo_string50:
	.asciz	"_wide_data"                    # string offset=731
.Linfo_string51:
	.asciz	"_IO_wide_data"                 # string offset=742
.Linfo_string52:
	.asciz	"_freeres_list"                 # string offset=756
.Linfo_string53:
	.asciz	"_freeres_buf"                  # string offset=770
.Linfo_string54:
	.asciz	"__pad5"                        # string offset=783
.Linfo_string55:
	.asciz	"_mode"                         # string offset=790
.Linfo_string56:
	.asciz	"_unused2"                      # string offset=796
.Linfo_string57:
	.asciz	"_IO_FILE"                      # string offset=805
.Linfo_string58:
	.asciz	"FILE"                          # string offset=814
.Linfo_string59:
	.asciz	"fopen"                         # string offset=819
.Linfo_string60:
	.asciz	"fclose"                        # string offset=825
.Linfo_string61:
	.asciz	"free"                          # string offset=832
.Linfo_string62:
	.asciz	"perror"                        # string offset=837
.Linfo_string63:
	.asciz	"main"                          # string offset=844
.Linfo_string64:
	.asciz	"start"                         # string offset=849
.Linfo_string65:
	.asciz	"end"                           # string offset=855
.Linfo_string66:
	.asciz	"argc"                          # string offset=859
.Linfo_string67:
	.asciz	"argv"                          # string offset=864
.Linfo_string68:
	.asciz	"a"                             # string offset=869
.Linfo_string69:
	.asciz	"double"                        # string offset=871
.Linfo_string70:
	.asciz	"b"                             # string offset=878
.Linfo_string71:
	.asciz	"c"                             # string offset=880
.Linfo_string72:
	.asciz	"i"                             # string offset=882
.Linfo_string73:
	.asciz	"time_taken"                    # string offset=884
.Linfo_string74:
	.asciz	"f"                             # string offset=895
.Linfo_string75:
	.asciz	"j"                             # string offset=897
.Linfo_string76:
	.asciz	"k"                             # string offset=899
	.ident	"Intel(R) oneAPI DPC++/C++ Compiler 2023.2.0 (2023.2.0.20230721)"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
