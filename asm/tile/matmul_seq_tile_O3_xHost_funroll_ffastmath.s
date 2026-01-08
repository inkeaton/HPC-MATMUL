	.text
	.file	"matmul_seq_tile.c"
	.file	1 "/home/Zsf/Desktop/_FINAL_PROJECT" "src/matmul_seq_tile.c"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function main
.LCPI0_0:
	.quad	0x4000000000000000              #  2
.LCPI0_1:
	.quad	0x4008000000000000              #  3
.LCPI0_3:
	.quad	0x3e112e0be826d695              #  1.0000000000000001E-9
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5, 0x0
.LCPI0_2:
	.quad	160000                          # 0x27100
	.quad	200000                          # 0x30d40
	.quad	240000                          # 0x3a980
	.quad	280000                          # 0x445c0
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # 
.Lfunc_begin0:
	.loc	1 46 0                          # src/matmul_seq_tile.c:46:0
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
	subq	$232, %rsp
	.cfi_def_cfa_offset 288
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
	.loc	1 54 32 prologue_end            # src/matmul_seq_tile.c:54:32
	movl	$64, %edi
	movl	$200000000, %esi                # imm = 0xBEBC200
	callq	aligned_alloc
.Ltmp3:
	movq	%rax, %rbx
.Ltmp4:
	#DEBUG_VALUE: main:a <- $rbx
	.loc	1 55 32                         # src/matmul_seq_tile.c:55:32
	movl	$64, %edi
	movl	$200000000, %esi                # imm = 0xBEBC200
	callq	aligned_alloc
.Ltmp5:
	movq	%rax, %r13
.Ltmp6:
	#DEBUG_VALUE: main:b <- $r13
	.loc	1 56 32                         # src/matmul_seq_tile.c:56:32
	movl	$64, %edi
	movl	$200000000, %esi                # imm = 0xBEBC200
	callq	aligned_alloc
.Ltmp7:
	#DEBUG_VALUE: main:c <- $rax
	.loc	1 58 10                         # src/matmul_seq_tile.c:58:10
	testq	%rbx, %rbx
	.loc	1 58 12 is_stmt 0               # src/matmul_seq_tile.c:58:12
	je	.LBB0_13
.Ltmp8:
# %bb.1:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r13
	#DEBUG_VALUE: main:c <- $rax
	testq	%r13, %r13
	je	.LBB0_13
.Ltmp9:
# %bb.2:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r13
	#DEBUG_VALUE: main:c <- $rax
	.loc	1 56 32 is_stmt 1               # src/matmul_seq_tile.c:56:32
	movq	%rax, %r14
.Ltmp10:
	.loc	1 58 12                         # src/matmul_seq_tile.c:58:12
	testq	%rax, %rax
	je	.LBB0_13
.Ltmp11:
# %bb.3:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r13
	#DEBUG_VALUE: main:c <- $r14
	.loc	1 70 21                         # src/matmul_seq_tile.c:70:21
	movl	$200000000, %edx                # imm = 0xBEBC200
	movq	%r14, %rdi
	xorl	%esi, %esi
	callq	_intel_fast_memset@PLT
.Ltmp12:
	.loc	1 0 21 is_stmt 0                # src/matmul_seq_tile.c:0:21
	movq	$-40, %rax
	.loc	1 68 21 is_stmt 1 discriminator 2 # src/matmul_seq_tile.c:68:21
	vbroadcastsd	.LCPI0_0(%rip), %ymm0   # ymm0 = [2.0E+0,2.0E+0,2.0E+0,2.0E+0]
	.loc	1 69 21 discriminator 2         # src/matmul_seq_tile.c:69:21
	vbroadcastsd	.LCPI0_1(%rip), %ymm1   # ymm1 = [3.0E+0,3.0E+0,3.0E+0,3.0E+0]
.Ltmp13:
	.p2align	4, 0x90
.LBB0_4:                                # =>This Inner Loop Header: Depth=1
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r13
	#DEBUG_VALUE: main:c <- $r14
	.loc	1 68 21                         # src/matmul_seq_tile.c:68:21
	vmovntpd	%ymm0, 320(%rbx,%rax,8)
	.loc	1 69 21                         # src/matmul_seq_tile.c:69:21
	vmovntpd	%ymm1, 320(%r13,%rax,8)
	.loc	1 68 21                         # src/matmul_seq_tile.c:68:21
	vmovntpd	%ymm0, 352(%rbx,%rax,8)
	.loc	1 69 21                         # src/matmul_seq_tile.c:69:21
	vmovntpd	%ymm1, 352(%r13,%rax,8)
	.loc	1 68 21                         # src/matmul_seq_tile.c:68:21
	vmovntpd	%ymm0, 384(%rbx,%rax,8)
	.loc	1 69 21                         # src/matmul_seq_tile.c:69:21
	vmovntpd	%ymm1, 384(%r13,%rax,8)
	.loc	1 68 21                         # src/matmul_seq_tile.c:68:21
	vmovntpd	%ymm0, 416(%rbx,%rax,8)
	.loc	1 69 21                         # src/matmul_seq_tile.c:69:21
	vmovntpd	%ymm1, 416(%r13,%rax,8)
	.loc	1 68 21                         # src/matmul_seq_tile.c:68:21
	vmovntpd	%ymm0, 448(%rbx,%rax,8)
	.loc	1 69 21                         # src/matmul_seq_tile.c:69:21
	vmovntpd	%ymm1, 448(%r13,%rax,8)
	.loc	1 68 21                         # src/matmul_seq_tile.c:68:21
	vmovntpd	%ymm0, 480(%rbx,%rax,8)
	.loc	1 69 21                         # src/matmul_seq_tile.c:69:21
	vmovntpd	%ymm1, 480(%r13,%rax,8)
	.loc	1 68 21                         # src/matmul_seq_tile.c:68:21
	vmovntpd	%ymm0, 512(%rbx,%rax,8)
	.loc	1 69 21                         # src/matmul_seq_tile.c:69:21
	vmovntpd	%ymm1, 512(%r13,%rax,8)
	.loc	1 68 21                         # src/matmul_seq_tile.c:68:21
	vmovntpd	%ymm0, 544(%rbx,%rax,8)
	.loc	1 69 21                         # src/matmul_seq_tile.c:69:21
	vmovntpd	%ymm1, 544(%r13,%rax,8)
	.loc	1 68 21                         # src/matmul_seq_tile.c:68:21
	vmovntpd	%ymm0, 576(%rbx,%rax,8)
	.loc	1 69 21                         # src/matmul_seq_tile.c:69:21
	vmovntpd	%ymm1, 576(%r13,%rax,8)
	.loc	1 68 21                         # src/matmul_seq_tile.c:68:21
	vmovntpd	%ymm0, 608(%rbx,%rax,8)
	.loc	1 69 21                         # src/matmul_seq_tile.c:69:21
	vmovntpd	%ymm1, 608(%r13,%rax,8)
.Ltmp14:
	.loc	1 66 27                         # src/matmul_seq_tile.c:66:27
	addq	$40, %rax
	cmpq	$24999960, %rax                 # imm = 0x17D7818
.Ltmp15:
	.loc	1 66 9 is_stmt 0                # src/matmul_seq_tile.c:66:9
	jb	.LBB0_4
.Ltmp16:
# %bb.5:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r13
	#DEBUG_VALUE: main:c <- $r14
	.loc	1 0 9                           # src/matmul_seq_tile.c:0:9
	movq	%rbx, 32(%rsp)                  # 8-byte Spill
.Ltmp17:
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	.loc	1 58 12 is_stmt 1               # src/matmul_seq_tile.c:58:12
	sfence
	leaq	216(%rsp), %rsi
.Ltmp18:
	.loc	1 76 9                          # src/matmul_seq_tile.c:76:9
	movl	$1, %edi
	vzeroupper
	callq	clock_gettime
.Ltmp19:
	#DEBUG_VALUE: ii <- 0
	.loc	1 0 9 is_stmt 0                 # src/matmul_seq_tile.c:0:9
	movq	%r13, %rax
	addq	$280000, %rax                   # imm = 0x445C0
	movq	%rax, 152(%rsp)                 # 8-byte Spill
	xorl	%ecx, %ecx
.Ltmp20:
	.loc	1 109 44 is_stmt 1 discriminator 2 # src/matmul_seq_tile.c:109:44
	vmovdqu	.LCPI0_2(%rip), %ymm9           # ymm9 = [160000,200000,240000,280000]
                                        # AlignMOV convert to UnAlignMOV 
	movq	%r14, 72(%rsp)                  # 8-byte Spill
.Ltmp21:
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	.loc	1 0 44 is_stmt 0                # src/matmul_seq_tile.c:0:44
	movq	%r14, 80(%rsp)                  # 8-byte Spill
	movq	%r13, 160(%rsp)                 # 8-byte Spill
.Ltmp22:
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	jmp	.LBB0_6
.Ltmp23:
	.p2align	4, 0x90
.LBB0_29:                               #   in Loop: Header=BB0_6 Depth=1
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	movq	168(%rsp), %rcx                 # 8-byte Reload
	.loc	1 83 25 is_stmt 1               # src/matmul_seq_tile.c:83:25
	leal	1(%rcx), %eax
.Ltmp24:
	.loc	1 83 5 is_stmt 0                # src/matmul_seq_tile.c:83:5
	addq	$2560000, 80(%rsp)              # 8-byte Folded Spill
                                        # imm = 0x271000
.Ltmp25:
	.loc	1 83 25                         # src/matmul_seq_tile.c:83:25
	cmpl	$78, %ecx
	movl	%eax, %ecx
	movq	160(%rsp), %r13                 # 8-byte Reload
.Ltmp26:
	.loc	1 83 5                          # src/matmul_seq_tile.c:83:5
	je	.LBB0_30
.Ltmp27:
.LBB0_6:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_7 Depth 2
                                        #       Child Loop BB0_9 Depth 3
                                        #         Child Loop BB0_40 Depth 4
                                        #           Child Loop BB0_38 Depth 5
                                        #             Child Loop BB0_36 Depth 6
                                        #             Child Loop BB0_33 Depth 6
                                        #         Child Loop BB0_17 Depth 4
                                        #           Child Loop BB0_19 Depth 5
                                        #             Child Loop BB0_24 Depth 6
                                        #             Child Loop BB0_22 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	.loc	1 43 39 is_stmt 1               # src/matmul_seq_tile.c:43:39
	movl	%ecx, %eax
	shll	$6, %eax
	movq	%rax, 184(%rsp)                 # 8-byte Spill
	leal	64(%rax), %edx
	.loc	1 43 42 is_stmt 0               # src/matmul_seq_tile.c:43:42
	cmpl	$5000, %edx                     # imm = 0x1388
	.loc	1 43 39                         # src/matmul_seq_tile.c:43:39
	movl	$5000, %eax                     # imm = 0x1388
	cmovael	%eax, %edx
.Ltmp28:
	#DEBUG_VALUE: i_max <- $edx
	.loc	1 0 39                          # src/matmul_seq_tile.c:0:39
	movq	%rcx, 168(%rsp)                 # 8-byte Spill
.Ltmp29:
	.loc	1 95 17 is_stmt 1               # src/matmul_seq_tile.c:95:17
	movl	%ecx, %eax
	shlq	$6, %rax
	movq	%rax, 120(%rsp)                 # 8-byte Spill
	movq	%rax, %rsi
	notq	%rsi
	movq	%rdx, 176(%rsp)                 # 8-byte Spill
.Ltmp30:
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	addq	%rdx, %rsi
	movq	%r13, 96(%rsp)                  # 8-byte Spill
	movq	152(%rsp), %rax                 # 8-byte Reload
	movq	%rax, 88(%rsp)                  # 8-byte Spill
	xorl	%eax, %eax
	movq	%rsi, 128(%rsp)                 # 8-byte Spill
	jmp	.LBB0_7
.Ltmp31:
	.p2align	4, 0x90
.LBB0_28:                               #   in Loop: Header=BB0_7 Depth=2
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	.loc	1 0 17 is_stmt 0                # src/matmul_seq_tile.c:0:17
	movq	104(%rsp), %rcx                 # 8-byte Reload
	.loc	1 84 29 is_stmt 1               # src/matmul_seq_tile.c:84:29
	leal	1(%rcx), %eax
.Ltmp32:
	.loc	1 84 9 is_stmt 0                # src/matmul_seq_tile.c:84:9
	addq	$2560000, 88(%rsp)              # 8-byte Folded Spill
                                        # imm = 0x271000
	addq	$2560000, 96(%rsp)              # 8-byte Folded Spill
                                        # imm = 0x271000
.Ltmp33:
	.loc	1 84 29                         # src/matmul_seq_tile.c:84:29
	cmpl	$78, %ecx
.Ltmp34:
                                        # kill: def $eax killed $eax def $rax
	.loc	1 84 9                          # src/matmul_seq_tile.c:84:9
	je	.LBB0_29
.Ltmp35:
.LBB0_7:                                #   Parent Loop BB0_6 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_9 Depth 3
                                        #         Child Loop BB0_40 Depth 4
                                        #           Child Loop BB0_38 Depth 5
                                        #             Child Loop BB0_36 Depth 6
                                        #             Child Loop BB0_33 Depth 6
                                        #         Child Loop BB0_17 Depth 4
                                        #           Child Loop BB0_19 Depth 5
                                        #             Child Loop BB0_24 Depth 6
                                        #             Child Loop BB0_22 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	.loc	1 0 9                           # src/matmul_seq_tile.c:0:9
	movq	%rax, 104(%rsp)                 # 8-byte Spill
                                        # kill: def $eax killed $eax killed $rax def $rax
.Ltmp36:
	.loc	1 43 39 is_stmt 1               # src/matmul_seq_tile.c:43:39
	shll	$6, %eax
	leal	64(%rax), %edx
	.loc	1 43 42 is_stmt 0               # src/matmul_seq_tile.c:43:42
	cmpl	$5000, %edx                     # imm = 0x1388
	.loc	1 43 39                         # src/matmul_seq_tile.c:43:39
	movl	$5000, %ecx                     # imm = 0x1388
	cmovael	%ecx, %edx
.Ltmp37:
	#DEBUG_VALUE: k_max <- $edx
	.loc	1 0 39                          # src/matmul_seq_tile.c:0:39
	movq	%rdx, 56(%rsp)                  # 8-byte Spill
.Ltmp38:
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	.loc	1 96 40 is_stmt 1               # src/matmul_seq_tile.c:96:40
	cmpl	%edx, %eax
.Ltmp39:
	.loc	1 96 21 is_stmt 0               # src/matmul_seq_tile.c:96:21
	jae	.LBB0_28
.Ltmp40:
# %bb.8:                                #   in Loop: Header=BB0_7 Depth=2
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	.loc	1 58 12 is_stmt 1               # src/matmul_seq_tile.c:58:12
	movl	104(%rsp), %eax                 # 4-byte Reload
	shlq	$6, %rax
	movq	56(%rsp), %rcx                  # 8-byte Reload
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	subq	%rax, %rcx
	shrq	$3, %rcx
.Ltmp41:
	.loc	1 95 17                         # src/matmul_seq_tile.c:95:17
	decq	%rcx
	movq	%rcx, 144(%rsp)                 # 8-byte Spill
	movq	$0, 112(%rsp)                   # 8-byte Folded Spill
	movl	$64, %ecx
	movq	96(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	88(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	80(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	movq	$0, 24(%rsp)                    # 8-byte Folded Spill
	jmp	.LBB0_9
.Ltmp42:
	.p2align	4, 0x90
.LBB0_27:                               #   in Loop: Header=BB0_9 Depth=3
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	.loc	1 0 17 is_stmt 0                # src/matmul_seq_tile.c:0:17
	movq	24(%rsp), %rdx                  # 8-byte Reload
	.loc	1 85 33 is_stmt 1               # src/matmul_seq_tile.c:85:33
	leal	1(%rdx), %eax
.Ltmp43:
	.loc	1 85 13 is_stmt 0               # src/matmul_seq_tile.c:85:13
	addq	$512, 16(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x200
	addq	$512, 40(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x200
	movq	192(%rsp), %rcx                 # 8-byte Reload
	addq	$64, %rcx
	addq	$-64, 112(%rsp)                 # 8-byte Folded Spill
	addq	$512, 48(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x200
.Ltmp44:
	.loc	1 85 33                         # src/matmul_seq_tile.c:85:33
	cmpl	$78, %edx
                                        # kill: def $eax killed $eax def $rax
	movq	%rax, 24(%rsp)                  # 8-byte Spill
.Ltmp45:
	.loc	1 85 13                         # src/matmul_seq_tile.c:85:13
	je	.LBB0_28
.Ltmp46:
.LBB0_9:                                #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_40 Depth 4
                                        #           Child Loop BB0_38 Depth 5
                                        #             Child Loop BB0_36 Depth 6
                                        #             Child Loop BB0_33 Depth 6
                                        #         Child Loop BB0_17 Depth 4
                                        #           Child Loop BB0_19 Depth 5
                                        #             Child Loop BB0_24 Depth 6
                                        #             Child Loop BB0_22 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	.loc	1 95 17 is_stmt 1               # src/matmul_seq_tile.c:95:17
	cmpq	$5000, %rcx                     # imm = 0x1388
	movl	$5000, %r15d                    # imm = 0x1388
	movq	%rcx, 192(%rsp)                 # 8-byte Spill
	cmovbq	%rcx, %r15
	movq	176(%rsp), %rax                 # 8-byte Reload
.Ltmp47:
	.loc	1 95 36 is_stmt 0               # src/matmul_seq_tile.c:95:36
	cmpl	%eax, 184(%rsp)                 # 4-byte Folded Reload
.Ltmp48:
	.loc	1 95 17                         # src/matmul_seq_tile.c:95:17
	jae	.LBB0_27
.Ltmp49:
# %bb.10:                               #   in Loop: Header=BB0_9 Depth=3
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	.loc	1 0 17                          # src/matmul_seq_tile.c:0:17
	movq	24(%rsp), %rcx                  # 8-byte Reload
.Ltmp50:
	.loc	1 43 39 is_stmt 1               # src/matmul_seq_tile.c:43:39
	movl	%ecx, %ebx
	shll	$6, %ebx
	leal	64(%rbx), %r12d
	.loc	1 43 42 is_stmt 0               # src/matmul_seq_tile.c:43:42
	cmpl	$5000, %r12d                    # imm = 0x1388
	.loc	1 43 39                         # src/matmul_seq_tile.c:43:39
	movl	$5000, %eax                     # imm = 0x1388
	cmovael	%eax, %r12d
.Ltmp51:
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 108 44 is_stmt 1              # src/matmul_seq_tile.c:108:44
	movl	%ecx, %r14d
	shlq	$6, %r14
	#APP
	rdpid	%rax
	#NO_APP
	andl	$1023, %eax                     # imm = 0x3FF
	movq	__cpu_core_type@GOTPCREL(%rip), %rcx
	movzbl	(%rcx,%rax), %eax
	testb	%al, %al
	je	.LBB0_47
.Ltmp52:
.LBB0_11:                               #   in Loop: Header=BB0_9 Depth=3
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 95 17                         # src/matmul_seq_tile.c:95:17
	addq	112(%rsp), %r15                 # 8-byte Folded Reload
.Ltmp53:
	.loc	1 108 44                        # src/matmul_seq_tile.c:108:44
	movq	%r12, %rcx
	movq	%r14, 136(%rsp)                 # 8-byte Spill
	subq	%r14, %rcx
	cmpb	$32, %al
	je	.LBB0_12
.Ltmp54:
# %bb.16:                               #   in Loop: Header=BB0_9 Depth=3
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 0 44 is_stmt 0                # src/matmul_seq_tile.c:0:44
	movq	16(%rsp), %rax                  # 8-byte Reload
	xorl	%ebp, %ebp
	jmp	.LBB0_17
.Ltmp55:
	.p2align	4, 0x90
.LBB0_26:                               #   in Loop: Header=BB0_17 Depth=4
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 95 17 is_stmt 1               # src/matmul_seq_tile.c:95:17
	addq	$40000, %rax                    # imm = 0x9C40
	movq	128(%rsp), %rsi                 # 8-byte Reload
.Ltmp56:
	.loc	1 95 36 is_stmt 0               # src/matmul_seq_tile.c:95:36
	cmpq	%rsi, %rbp
	leaq	1(%rbp), %rbp
.Ltmp57:
	.loc	1 95 17                         # src/matmul_seq_tile.c:95:17
	je	.LBB0_27
.Ltmp58:
.LBB0_17:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_9 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_19 Depth 5
                                        #             Child Loop BB0_24 Depth 6
                                        #             Child Loop BB0_22 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 0 17                          # src/matmul_seq_tile.c:0:17
	movq	8(%rsp), %rdx                   # 8-byte Reload
	.loc	1 95 17                         # src/matmul_seq_tile.c:95:17
	cmpq	56(%rsp), %rdx                  # 8-byte Folded Reload
	je	.LBB0_26
.Ltmp59:
# %bb.18:                               #   in Loop: Header=BB0_17 Depth=4
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 0 17                          # src/matmul_seq_tile.c:0:17
	movq	120(%rsp), %rdx                 # 8-byte Reload
.Ltmp60:
	.loc	1 99 36 is_stmt 1               # src/matmul_seq_tile.c:99:36
	leaq	(%rdx,%rbp), %rsi
	.loc	1 99 36 is_stmt 0 discriminator 2 # src/matmul_seq_tile.c:99:36
	imulq	$40000, %rsi, %rsi              # imm = 0x9C40
	.loc	1 99 36 discriminator 4         # src/matmul_seq_tile.c:99:36
	addq	32(%rsp), %rsi                  # 8-byte Folded Reload
	movq	48(%rsp), %rdi                  # 8-byte Reload
	movq	40(%rsp), %r8                   # 8-byte Reload
	xorl	%r9d, %r9d
	jmp	.LBB0_19
.Ltmp61:
	.p2align	4, 0x90
.LBB0_25:                               #   in Loop: Header=BB0_19 Depth=5
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 96 21 is_stmt 1               # src/matmul_seq_tile.c:96:21
	addq	$320000, %r8                    # imm = 0x4E200
	addq	$320000, %rdi                   # imm = 0x4E200
.Ltmp62:
	.loc	1 96 40 is_stmt 0               # src/matmul_seq_tile.c:96:40
	cmpq	144(%rsp), %r9                  # 8-byte Folded Reload
	leaq	1(%r9), %r9
.Ltmp63:
	.loc	1 96 21                         # src/matmul_seq_tile.c:96:21
	je	.LBB0_26
.Ltmp64:
.LBB0_19:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_9 Depth=3
                                        #         Parent Loop BB0_17 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_24 Depth 6
                                        #             Child Loop BB0_22 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 108 44 is_stmt 1              # src/matmul_seq_tile.c:108:44
	cmpl	%r12d, %ebx
.Ltmp65:
	#DEBUG_VALUE: r <- undef
	.loc	1 108 25 is_stmt 0              # src/matmul_seq_tile.c:108:25
	jae	.LBB0_25
.Ltmp66:
# %bb.20:                               #   in Loop: Header=BB0_19 Depth=5
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 0 25                          # src/matmul_seq_tile.c:0:25
	movq	8(%rsp), %rdx                   # 8-byte Reload
	.loc	1 99 36 is_stmt 1               # src/matmul_seq_tile.c:99:36
	leaq	(%rdx,%r9,8), %r10
	vmovupd	(%rsi,%r10,8), %ymm1            # AlignMOV convert to UnAlignMOV 
	vmovupd	32(%rsi,%r10,8), %ymm0          # AlignMOV convert to UnAlignMOV 
.Ltmp67:
	.loc	1 58 12                         # src/matmul_seq_tile.c:58:12
	cmpq	%r12, 136(%rsp)                 # 8-byte Folded Reload
.Ltmp68:
	.loc	1 108 25                        # src/matmul_seq_tile.c:108:25
	jne	.LBB0_23
.Ltmp69:
# %bb.21:                               #   in Loop: Header=BB0_19 Depth=5
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 0 25 is_stmt 0                # src/matmul_seq_tile.c:0:25
	movq	%rdi, %r10
	xorl	%r11d, %r11d
.Ltmp70:
	.p2align	4, 0x90
.LBB0_22:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_9 Depth=3
                                        #         Parent Loop BB0_17 Depth=4
                                        #           Parent Loop BB0_19 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 109 44 is_stmt 1              # src/matmul_seq_tile.c:109:44
	leaq	40000(%r10), %r14
	leaq	80000(%r10), %rdx
	vxorpd	%xmm2, %xmm2, %xmm2
	vpcmpeqd	%ymm3, %ymm3, %ymm3
	vgatherqpd	%ymm3, (%r10,%ymm9), %ymm2
	leaq	120000(%r10), %r13
	vmovq	%r13, %xmm3
	vmovq	%rdx, %xmm4
	vpunpcklqdq	%xmm3, %xmm4, %xmm3     # xmm3 = xmm4[0],xmm3[0]
	vmovq	%r10, %xmm4
	vmovq	%r14, %xmm5
	vpunpcklqdq	%xmm5, %xmm4, %xmm4     # xmm4 = xmm4[0],xmm5[0]
	vinserti128	$1, %xmm3, %ymm4, %ymm3
	vpxor	%xmm4, %xmm4, %xmm4
	vpcmpeqd	%ymm5, %ymm5, %ymm5
	vgatherqpd	%ymm5, (,%ymm3), %ymm4
	.loc	1 109 42 is_stmt 0              # src/matmul_seq_tile.c:109:42
	vmulpd	%ymm0, %ymm2, %ymm2
	.loc	1 109 37                        # src/matmul_seq_tile.c:109:37
	vfmadd231pd	%ymm4, %ymm1, %ymm2     # ymm2 = (ymm1 * ymm4) + ymm2
	vextractf128	$1, %ymm2, %xmm3
	vaddpd	%xmm3, %xmm2, %xmm2
	vpermilpd	$1, %xmm2, %xmm3        # xmm3 = xmm2[1,0]
	vaddsd	%xmm3, %xmm2, %xmm2
	vaddsd	(%rax,%r11,8), %xmm2, %xmm2
	vmovsd	%xmm2, (%rax,%r11,8)
.Ltmp71:
	.loc	1 108 44 is_stmt 1              # src/matmul_seq_tile.c:108:44
	incq	%r11
	addq	$8, %r10
	cmpq	%r11, %r15
	jne	.LBB0_22
	jmp	.LBB0_25
.Ltmp72:
	.p2align	4, 0x90
.LBB0_23:                               #   in Loop: Header=BB0_19 Depth=5
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 109 42                        # src/matmul_seq_tile.c:109:42
	vbroadcastsd	%xmm1, %ymm2
	vxorps	%xmm3, %xmm3, %xmm3
	vpermpd	$85, %ymm1, %ymm3               # ymm3 = ymm1[1,1,1,1]
	vxorps	%xmm4, %xmm4, %xmm4
	vpermpd	$170, %ymm1, %ymm4              # ymm4 = ymm1[2,2,2,2]
	vpermpd	$255, %ymm1, %ymm1              # ymm1 = ymm1[3,3,3,3]
	vbroadcastsd	%xmm0, %ymm5
	vxorps	%xmm6, %xmm6, %xmm6
	vpermpd	$85, %ymm0, %ymm6               # ymm6 = ymm0[1,1,1,1]
	vxorps	%xmm7, %xmm7, %xmm7
	vpermpd	$170, %ymm0, %ymm7              # ymm7 = ymm0[2,2,2,2]
	vpermpd	$255, %ymm0, %ymm0              # ymm0 = ymm0[3,3,3,3]
	xorl	%r10d, %r10d
.Ltmp73:
	.p2align	4, 0x90
.LBB0_24:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_9 Depth=3
                                        #         Parent Loop BB0_17 Depth=4
                                        #           Parent Loop BB0_19 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 109 44 is_stmt 0              # src/matmul_seq_tile.c:109:44
	vmovupd	-280000(%r8,%r10,8), %ymm8      # AlignMOV convert to UnAlignMOV 
	.loc	1 109 37                        # src/matmul_seq_tile.c:109:37
	vfmadd213pd	(%rax,%r10,8), %ymm2, %ymm8 # ymm8 = (ymm2 * ymm8) + mem
	vfmadd231pd	-240000(%r8,%r10,8), %ymm3, %ymm8 # ymm8 = (ymm3 * mem) + ymm8
	vfmadd231pd	-200000(%r8,%r10,8), %ymm4, %ymm8 # ymm8 = (ymm4 * mem) + ymm8
	vfmadd231pd	-160000(%r8,%r10,8), %ymm1, %ymm8 # ymm8 = (ymm1 * mem) + ymm8
	vfmadd231pd	-120000(%r8,%r10,8), %ymm5, %ymm8 # ymm8 = (ymm5 * mem) + ymm8
	vfmadd231pd	-80000(%r8,%r10,8), %ymm6, %ymm8 # ymm8 = (ymm6 * mem) + ymm8
	vfmadd231pd	-40000(%r8,%r10,8), %ymm7, %ymm8 # ymm8 = (ymm7 * mem) + ymm8
	vfmadd231pd	(%r8,%r10,8), %ymm0, %ymm8 # ymm8 = (ymm0 * mem) + ymm8
	vmovupd	%ymm8, (%rax,%r10,8)            # AlignMOV convert to UnAlignMOV 
.Ltmp74:
	.loc	1 108 44 is_stmt 1              # src/matmul_seq_tile.c:108:44
	addq	$4, %r10
	cmpq	%rcx, %r10
.Ltmp75:
	.loc	1 108 25 is_stmt 0              # src/matmul_seq_tile.c:108:25
	jl	.LBB0_24
	jmp	.LBB0_25
.Ltmp76:
	.p2align	4, 0x90
.LBB0_12:                               #   in Loop: Header=BB0_9 Depth=3
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 0 25                          # src/matmul_seq_tile.c:0:25
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	$0, 64(%rsp)                    # 8-byte Folded Spill
	jmp	.LBB0_40
.Ltmp77:
	.p2align	4, 0x90
.LBB0_39:                               #   in Loop: Header=BB0_40 Depth=4
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 95 17 is_stmt 1               # src/matmul_seq_tile.c:95:17
	addq	$40000, %rax                    # imm = 0x9C40
	movq	128(%rsp), %rsi                 # 8-byte Reload
	movq	64(%rsp), %rdx                  # 8-byte Reload
.Ltmp78:
	.loc	1 95 36 is_stmt 0               # src/matmul_seq_tile.c:95:36
	cmpq	%rsi, %rdx
	leaq	1(%rdx), %rdx
	movq	%rdx, 64(%rsp)                  # 8-byte Spill
.Ltmp79:
	.loc	1 95 17                         # src/matmul_seq_tile.c:95:17
	je	.LBB0_27
.Ltmp80:
.LBB0_40:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_9 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_38 Depth 5
                                        #             Child Loop BB0_36 Depth 6
                                        #             Child Loop BB0_33 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 0 17                          # src/matmul_seq_tile.c:0:17
	movq	8(%rsp), %rdx                   # 8-byte Reload
	.loc	1 95 17                         # src/matmul_seq_tile.c:95:17
	cmpq	56(%rsp), %rdx                  # 8-byte Folded Reload
	je	.LBB0_39
.Ltmp81:
# %bb.41:                               #   in Loop: Header=BB0_40 Depth=4
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 0 17                          # src/matmul_seq_tile.c:0:17
	movq	120(%rsp), %rdx                 # 8-byte Reload
	movq	64(%rsp), %rsi                  # 8-byte Reload
.Ltmp82:
	.loc	1 99 36 is_stmt 1               # src/matmul_seq_tile.c:99:36
	addq	%rsi, %rdx
	.loc	1 99 36 is_stmt 0 discriminator 6 # src/matmul_seq_tile.c:99:36
	imulq	$40000, %rdx, %rsi              # imm = 0x9C40
	.loc	1 99 36 discriminator 8         # src/matmul_seq_tile.c:99:36
	addq	32(%rsp), %rsi                  # 8-byte Folded Reload
	movq	48(%rsp), %rdi                  # 8-byte Reload
	movq	40(%rsp), %r8                   # 8-byte Reload
	xorl	%r9d, %r9d
	jmp	.LBB0_38
.Ltmp83:
	.p2align	4, 0x90
.LBB0_37:                               #   in Loop: Header=BB0_38 Depth=5
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 96 21 is_stmt 1               # src/matmul_seq_tile.c:96:21
	addq	$320000, %r8                    # imm = 0x4E200
	addq	$320000, %rdi                   # imm = 0x4E200
.Ltmp84:
	.loc	1 96 40 is_stmt 0               # src/matmul_seq_tile.c:96:40
	cmpq	144(%rsp), %r9                  # 8-byte Folded Reload
	leaq	1(%r9), %r9
.Ltmp85:
	.loc	1 96 21                         # src/matmul_seq_tile.c:96:21
	je	.LBB0_39
.Ltmp86:
.LBB0_38:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_9 Depth=3
                                        #         Parent Loop BB0_40 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_36 Depth 6
                                        #             Child Loop BB0_33 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 108 44 is_stmt 1              # src/matmul_seq_tile.c:108:44
	cmpl	%r12d, %ebx
.Ltmp87:
	#DEBUG_VALUE: r <- undef
	.loc	1 0 44 is_stmt 0                # src/matmul_seq_tile.c:0:44
	jae	.LBB0_37
.Ltmp88:
# %bb.34:                               #   in Loop: Header=BB0_38 Depth=5
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	movq	8(%rsp), %rdx                   # 8-byte Reload
	.loc	1 99 36 is_stmt 1               # src/matmul_seq_tile.c:99:36
	leaq	(%rdx,%r9,8), %rdx
	vmovupd	(%rsi,%rdx,8), %ymm1            # AlignMOV convert to UnAlignMOV 
	vmovupd	32(%rsi,%rdx,8), %ymm0          # AlignMOV convert to UnAlignMOV 
.Ltmp89:
	.loc	1 58 12                         # src/matmul_seq_tile.c:58:12
	cmpq	%r12, 136(%rsp)                 # 8-byte Folded Reload
.Ltmp90:
	.loc	1 108 25                        # src/matmul_seq_tile.c:108:25
	jne	.LBB0_35
.Ltmp91:
# %bb.32:                               #   in Loop: Header=BB0_38 Depth=5
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 0 25 is_stmt 0                # src/matmul_seq_tile.c:0:25
	movq	%rdi, %r10
	xorl	%r11d, %r11d
.Ltmp92:
	.p2align	4, 0x90
.LBB0_33:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_9 Depth=3
                                        #         Parent Loop BB0_40 Depth=4
                                        #           Parent Loop BB0_38 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 109 44 is_stmt 1              # src/matmul_seq_tile.c:109:44
	vmovq	%r10, %xmm2
	vpbroadcastq	%xmm2, %ymm2
	vpaddq	%ymm2, %ymm9, %ymm2
	vpextrq	$1, %xmm2, %r14
	vmovq	%xmm2, %r13
	vextracti128	$1, %ymm2, %xmm2
	vmovq	%xmm2, %rdx
	vpextrq	$1, %xmm2, %rbp
	vmovsd	80000(%r10), %xmm2              # xmm2 = mem[0],zero
	vmovhpd	120000(%r10), %xmm2, %xmm2      # xmm2 = xmm2[0],mem[0]
	vmovsd	(%r10), %xmm3                   # xmm3 = mem[0],zero
	vmovhpd	40000(%r10), %xmm3, %xmm3       # xmm3 = xmm3[0],mem[0]
	vinsertf128	$1, %xmm2, %ymm3, %ymm2
	vmovsd	(%rdx), %xmm3                   # xmm3 = mem[0],zero
	vmovhpd	(%rbp), %xmm3, %xmm3            # xmm3 = xmm3[0],mem[0]
	vmovsd	(%r13), %xmm4                   # xmm4 = mem[0],zero
	vmovhpd	(%r14), %xmm4, %xmm4            # xmm4 = xmm4[0],mem[0]
	vinsertf128	$1, %xmm3, %ymm4, %ymm3
	.loc	1 109 42 is_stmt 0              # src/matmul_seq_tile.c:109:42
	vmulpd	%ymm0, %ymm3, %ymm3
	.loc	1 109 37                        # src/matmul_seq_tile.c:109:37
	vfmadd231pd	%ymm2, %ymm1, %ymm3     # ymm3 = (ymm1 * ymm2) + ymm3
	vextractf128	$1, %ymm3, %xmm2
	vaddpd	%xmm2, %xmm3, %xmm2
	vpermilpd	$1, %xmm2, %xmm3        # xmm3 = xmm2[1,0]
	vaddsd	%xmm3, %xmm2, %xmm2
	vaddsd	(%rax,%r11,8), %xmm2, %xmm2
	vmovsd	%xmm2, (%rax,%r11,8)
.Ltmp93:
	.loc	1 108 44 is_stmt 1              # src/matmul_seq_tile.c:108:44
	incq	%r11
	addq	$8, %r10
	cmpq	%r11, %r15
.Ltmp94:
	.loc	1 108 25 is_stmt 0              # src/matmul_seq_tile.c:108:25
	jne	.LBB0_33
	jmp	.LBB0_37
.Ltmp95:
	.p2align	4, 0x90
.LBB0_35:                               #   in Loop: Header=BB0_38 Depth=5
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 109 42 is_stmt 1              # src/matmul_seq_tile.c:109:42
	vbroadcastsd	%xmm1, %ymm2
	vxorps	%xmm3, %xmm3, %xmm3
	vpermpd	$85, %ymm1, %ymm3               # ymm3 = ymm1[1,1,1,1]
	vxorps	%xmm4, %xmm4, %xmm4
	vpermpd	$170, %ymm1, %ymm4              # ymm4 = ymm1[2,2,2,2]
	vpermpd	$255, %ymm1, %ymm1              # ymm1 = ymm1[3,3,3,3]
	vbroadcastsd	%xmm0, %ymm5
	vxorps	%xmm6, %xmm6, %xmm6
	vpermpd	$85, %ymm0, %ymm6               # ymm6 = ymm0[1,1,1,1]
	vxorps	%xmm7, %xmm7, %xmm7
	vpermpd	$170, %ymm0, %ymm7              # ymm7 = ymm0[2,2,2,2]
	vpermpd	$255, %ymm0, %ymm0              # ymm0 = ymm0[3,3,3,3]
	xorl	%r10d, %r10d
.Ltmp96:
	.p2align	4, 0x90
.LBB0_36:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_9 Depth=3
                                        #         Parent Loop BB0_40 Depth=4
                                        #           Parent Loop BB0_38 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 109 44 is_stmt 0              # src/matmul_seq_tile.c:109:44
	vmovupd	-280000(%r8,%r10,8), %ymm8      # AlignMOV convert to UnAlignMOV 
	.loc	1 109 37                        # src/matmul_seq_tile.c:109:37
	vfmadd213pd	(%rax,%r10,8), %ymm2, %ymm8 # ymm8 = (ymm2 * ymm8) + mem
	vfmadd231pd	-240000(%r8,%r10,8), %ymm3, %ymm8 # ymm8 = (ymm3 * mem) + ymm8
	vfmadd231pd	-200000(%r8,%r10,8), %ymm4, %ymm8 # ymm8 = (ymm4 * mem) + ymm8
	vfmadd231pd	-160000(%r8,%r10,8), %ymm1, %ymm8 # ymm8 = (ymm1 * mem) + ymm8
	vfmadd231pd	-120000(%r8,%r10,8), %ymm5, %ymm8 # ymm8 = (ymm5 * mem) + ymm8
	vfmadd231pd	-80000(%r8,%r10,8), %ymm6, %ymm8 # ymm8 = (ymm6 * mem) + ymm8
	vfmadd231pd	-40000(%r8,%r10,8), %ymm7, %ymm8 # ymm8 = (ymm7 * mem) + ymm8
	vfmadd231pd	(%r8,%r10,8), %ymm0, %ymm8 # ymm8 = (ymm0 * mem) + ymm8
	vmovupd	%ymm8, (%rax,%r10,8)            # AlignMOV convert to UnAlignMOV 
.Ltmp97:
	.loc	1 108 44 is_stmt 1              # src/matmul_seq_tile.c:108:44
	addq	$4, %r10
	cmpq	%rcx, %r10
.Ltmp98:
	.loc	1 108 25 is_stmt 0              # src/matmul_seq_tile.c:108:25
	jl	.LBB0_36
	jmp	.LBB0_37
.Ltmp99:
.LBB0_47:                               #   in Loop: Header=BB0_9 Depth=3
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: i_max <- [DW_OP_plus_uconst 176, DW_OP_deref] $rsp
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 56, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r12d
	.loc	1 0 25                          # src/matmul_seq_tile.c:0:25
	vzeroupper
	callq	__detect_cpu_core_type@PLT
.Ltmp100:
	vmovdqu	.LCPI0_2(%rip), %ymm9           # ymm9 = [160000,200000,240000,280000]
                                        # AlignMOV convert to UnAlignMOV 
	jmp	.LBB0_11
.Ltmp101:
.LBB0_13:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r13
	#DEBUG_VALUE: main:c <- $rax
	.loc	1 59 9 is_stmt 1                # src/matmul_seq_tile.c:59:9
	movl	$.L.str, %edi
.Ltmp102:
.LBB0_14:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:b <- $r13
	.loc	1 0 0 is_stmt 0                 # src/matmul_seq_tile.c:0:0
	callq	perror
.Ltmp103:
	movl	$1, %eax
	jmp	.LBB0_15
.Ltmp104:
.LBB0_30:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	leaq	200(%rsp), %rsi
	.loc	1 119 9 is_stmt 1               # src/matmul_seq_tile.c:119:9
	movl	$1, %edi
	vzeroupper
	callq	clock_gettime
.Ltmp105:
	.loc	1 120 34                        # src/matmul_seq_tile.c:120:34
	movq	200(%rsp), %rax
	.loc	1 120 64 is_stmt 0              # src/matmul_seq_tile.c:120:64
	movq	208(%rsp), %rcx
	.loc	1 120 41                        # src/matmul_seq_tile.c:120:41
	subq	216(%rsp), %rax
	.loc	1 120 29                        # src/matmul_seq_tile.c:120:29
	vcvtsi2sd	%rax, %xmm10, %xmm1
	.loc	1 120 72                        # src/matmul_seq_tile.c:120:72
	subq	224(%rsp), %rcx
	.loc	1 120 59                        # src/matmul_seq_tile.c:120:59
	vcvtsi2sd	%rcx, %xmm10, %xmm0
	.loc	1 120 57                        # src/matmul_seq_tile.c:120:57
	vfmadd132sd	.LCPI0_3(%rip), %xmm1, %xmm0 # xmm0 = (xmm0 * mem) + xmm1
.Ltmp106:
	#DEBUG_VALUE: main:time_taken <- $xmm0
	.loc	1 122 17 is_stmt 1              # src/matmul_seq_tile.c:122:17
	movq	stderr(%rip), %rdi
	.loc	1 122 9 is_stmt 0               # src/matmul_seq_tile.c:122:9
	movl	$.L.str.1, %esi
	movl	$5000, %edx                     # imm = 0x1388
	movl	$64, %ecx
	movb	$1, %al
	callq	fprintf
.Ltmp107:
	.loc	1 127 15 is_stmt 1              # src/matmul_seq_tile.c:127:15
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %esi
	callq	fopen
.Ltmp108:
	#DEBUG_VALUE: main:f <- $rax
	.loc	1 128 10                        # src/matmul_seq_tile.c:128:10
	testq	%rax, %rax
.Ltmp109:
	.loc	1 128 9 is_stmt 0               # src/matmul_seq_tile.c:128:9
	je	.LBB0_31
.Ltmp110:
# %bb.42:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $rax
	.loc	1 127 15 is_stmt 1              # src/matmul_seq_tile.c:127:15
	movq	%rax, %r12
	xorl	%ebx, %ebx
	.loc	1 134 5                         # src/matmul_seq_tile.c:134:5
	movl	$.L.str.5, %esi
	movq	%rax, %rdi
	movl	$5000, %edx                     # imm = 0x1388
	xorl	%eax, %eax
.Ltmp111:
	#DEBUG_VALUE: main:f <- $r12
	callq	fprintf
.Ltmp112:
	#DEBUG_VALUE: i <- 0
	.loc	1 0 5 is_stmt 0                 # src/matmul_seq_tile.c:0:5
	movq	72(%rsp), %r14                  # 8-byte Reload
	addq	$56, %r14
.Ltmp113:
	.p2align	4, 0x90
.LBB0_43:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_44 Depth 2
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	#DEBUG_VALUE: i <- 0
	xorl	%r15d, %r15d
.Ltmp114:
	.p2align	4, 0x90
.LBB0_44:                               #   Parent Loop BB0_43 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	#DEBUG_VALUE: i <- 0
	.loc	1 139 33 is_stmt 1              # src/matmul_seq_tile.c:139:33
	vmovsd	-56(%r14,%r15), %xmm0           # xmm0 = mem[0],zero
	.loc	1 139 13 is_stmt 0              # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp115:
	.loc	1 139 33                        # src/matmul_seq_tile.c:139:33
	vmovsd	-48(%r14,%r15), %xmm0           # xmm0 = mem[0],zero
	.loc	1 139 13                        # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp116:
	.loc	1 139 33                        # src/matmul_seq_tile.c:139:33
	vmovsd	-40(%r14,%r15), %xmm0           # xmm0 = mem[0],zero
	.loc	1 139 13                        # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp117:
	.loc	1 139 33                        # src/matmul_seq_tile.c:139:33
	vmovsd	-32(%r14,%r15), %xmm0           # xmm0 = mem[0],zero
	.loc	1 139 13                        # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp118:
	.loc	1 139 33                        # src/matmul_seq_tile.c:139:33
	vmovsd	-24(%r14,%r15), %xmm0           # xmm0 = mem[0],zero
	.loc	1 139 13                        # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp119:
	.loc	1 139 33                        # src/matmul_seq_tile.c:139:33
	vmovsd	-16(%r14,%r15), %xmm0           # xmm0 = mem[0],zero
	.loc	1 139 13                        # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp120:
	.loc	1 139 33                        # src/matmul_seq_tile.c:139:33
	vmovsd	-8(%r14,%r15), %xmm0            # xmm0 = mem[0],zero
	.loc	1 139 13                        # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp121:
	.loc	1 139 33                        # src/matmul_seq_tile.c:139:33
	vmovsd	(%r14,%r15), %xmm0              # xmm0 = mem[0],zero
	.loc	1 139 13                        # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp122:
	.loc	1 137 27 is_stmt 1              # src/matmul_seq_tile.c:137:27
	addq	$64, %r15
	cmpq	$8000, %r15                     # imm = 0x1F40
.Ltmp123:
	.loc	1 137 9 is_stmt 0               # src/matmul_seq_tile.c:137:9
	jne	.LBB0_44
.Ltmp124:
# %bb.45:                               #   in Loop: Header=BB0_43 Depth=1
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	#DEBUG_VALUE: i <- 0
	.loc	1 141 9 is_stmt 1               # src/matmul_seq_tile.c:141:9
	movl	$10, %edi
	movq	%r12, %rsi
	callq	fputc@PLT
.Ltmp125:
	.loc	1 135 5                         # src/matmul_seq_tile.c:135:5
	addq	$40000, %r14                    # imm = 0x9C40
.Ltmp126:
	.loc	1 135 23 is_stmt 0              # src/matmul_seq_tile.c:135:23
	cmpq	$999, %rbx                      # imm = 0x3E7
	leaq	1(%rbx), %rbx
.Ltmp127:
	.loc	1 135 5                         # src/matmul_seq_tile.c:135:5
	jne	.LBB0_43
.Ltmp128:
# %bb.46:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	.loc	1 144 5 is_stmt 1               # src/matmul_seq_tile.c:144:5
	movq	%r12, %rdi
	callq	fclose
.Ltmp129:
	.loc	1 0 5 is_stmt 0                 # src/matmul_seq_tile.c:0:5
	movq	32(%rsp), %rdi                  # 8-byte Reload
	.loc	1 147 5 is_stmt 1               # src/matmul_seq_tile.c:147:5
	callq	free
.Ltmp130:
	.loc	1 148 5                         # src/matmul_seq_tile.c:148:5
	movq	%r13, %rdi
	callq	free
.Ltmp131:
	.loc	1 0 5 is_stmt 0                 # src/matmul_seq_tile.c:0:5
	movq	72(%rsp), %rdi                  # 8-byte Reload
	.loc	1 149 5 is_stmt 1               # src/matmul_seq_tile.c:149:5
	callq	free
.Ltmp132:
	.loc	1 0 5 is_stmt 0                 # src/matmul_seq_tile.c:0:5
	xorl	%eax, %eax
.Ltmp133:
.LBB0_15:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:b <- $r13
	.loc	1 151 1 epilogue_begin is_stmt 1 # src/matmul_seq_tile.c:151:1
	addq	$232, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
.Ltmp134:
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Ltmp135:
.LBB0_31:
	.cfi_def_cfa_offset 288
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 160, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 72, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $rax
	.loc	1 130 9                         # src/matmul_seq_tile.c:130:9
	movl	$.L.str.4, %edi
	jmp	.LBB0_14
.Ltmp136:
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
	.asciz	"[seq-tile] N=%d, block=%d, elapsed=%.3f s\n"
	.size	.L.str.1, 43

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
	.quad	.Ltmp17-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	83                              # DW_OP_reg3
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.short	2                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	32                              # 32
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	83                              # DW_OP_reg3
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp133-.Lfunc_begin0
	.short	2                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	32                              # 32
	.quad	.Ltmp135-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	2                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	32                              # 32
	.quad	0
	.quad	0
.Ldebug_loc3:
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	93                              # DW_OP_reg13
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	160                             # 160
	.byte	1                               # 
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	93                              # DW_OP_reg13
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp133-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	160                             # 160
	.byte	1                               # 
	.quad	.Ltmp133-.Lfunc_begin0
	.quad	.Ltmp134-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	93                              # DW_OP_reg13
	.quad	.Ltmp135-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	160                             # 160
	.byte	1                               # 
	.quad	0
	.quad	0
.Ldebug_loc4:
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	80                              # DW_OP_reg0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	94                              # DW_OP_reg14
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	200                             # 72
	.byte	0                               # 
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	80                              # DW_OP_reg0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp133-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	200                             # 72
	.byte	0                               # 
	.quad	.Ltmp135-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	200                             # 72
	.byte	0                               # 
	.quad	0
	.quad	0
.Ldebug_loc5:
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	81                              # super-register DW_OP_reg1
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	176                             # 176
	.byte	1                               # 
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	176                             # 176
	.byte	1                               # 
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	176                             # 176
	.byte	1                               # 
	.quad	0
	.quad	0
.Ldebug_loc6:
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	81                              # super-register DW_OP_reg1
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.short	2                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	56                              # 56
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.short	2                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	56                              # 56
	.quad	0
	.quad	0
.Ldebug_loc7:
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	92                              # super-register DW_OP_reg12
	.quad	0
	.quad	0
.Ldebug_loc8:
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	97                              # DW_OP_reg17
	.quad	0
	.quad	0
.Ldebug_loc9:
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	80                              # DW_OP_reg0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp133-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	92                              # DW_OP_reg12
	.quad	.Ltmp135-.Lfunc_begin0
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
	.byte	32                              # DW_AT_inline
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
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
	.byte	10                              # Abbreviation Code
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
	.byte	11                              # Abbreviation Code
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
	.byte	12                              # Abbreviation Code
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
	.byte	13                              # Abbreviation Code
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
	.byte	14                              # Abbreviation Code
	.byte	11                              # DW_TAG_lexical_block
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	15                              # Abbreviation Code
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
	.byte	16                              # Abbreviation Code
	.byte	11                              # DW_TAG_lexical_block
	.byte	1                               # DW_CHILDREN_yes
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	17                              # Abbreviation Code
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
	.byte	18                              # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	19                              # Abbreviation Code
	.ascii	"\211\202\001"                  # DW_TAG_GNU_call_site
	.byte	1                               # DW_CHILDREN_yes
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	20                              # Abbreviation Code
	.ascii	"\212\202\001"                  # DW_TAG_GNU_call_site_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\221B"                         # DW_AT_GNU_call_site_value
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	21                              # Abbreviation Code
	.ascii	"\211\202\001"                  # DW_TAG_GNU_call_site
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	22                              # Abbreviation Code
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
	.byte	23                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	24                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	25                              # Abbreviation Code
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
	.byte	26                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	27                              # Abbreviation Code
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
	.byte	28                              # Abbreviation Code
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
	.byte	29                              # Abbreviation Code
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
	.byte	30                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	31                              # Abbreviation Code
	.byte	24                              # DW_TAG_unspecified_parameters
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	32                              # Abbreviation Code
	.byte	55                              # DW_TAG_restrict_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	33                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	34                              # Abbreviation Code
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
	.byte	35                              # Abbreviation Code
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
	.byte	36                              # Abbreviation Code
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
	.byte	1                               # Abbrev [1] 0xb:0x7eb DW_TAG_compile_unit
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
	.byte	59                              # DW_AT_decl_line
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
	.byte	122                             # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	.L.str.1
	.byte	3                               # Abbrev [3] 0x6a:0xc DW_TAG_array_type
	.long	75                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x6f:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	43                              # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	2                               # Abbrev [2] 0x76:0x11 DW_TAG_variable
	.long	135                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	127                             # DW_AT_decl_line
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
	.byte	127                             # DW_AT_decl_line
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
	.byte	130                             # DW_AT_decl_line
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
	.byte	134                             # DW_AT_decl_line
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
	.byte	139                             # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	.L.str.6
	.byte	7                               # Abbrev [7] 0xfb:0x7 DW_TAG_variable
	.long	164                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	141                             # DW_AT_decl_line
	.byte	8                               # Abbrev [8] 0x102:0x23 DW_TAG_subprogram
	.long	.Linfo_string6                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	293                             # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_inline
	.byte	9                               # Abbrev [9] 0x10e:0xb DW_TAG_formal_parameter
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0x119:0xb DW_TAG_formal_parameter
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x125:0x7 DW_TAG_base_type
	.long	.Linfo_string7                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	10                              # Abbrev [10] 0x12c:0x3b7 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string66                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	45                              # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	293                             # DW_AT_type
                                        # DW_AT_external
	.byte	11                              # Abbrev [11] 0x145:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc0                    # DW_AT_location
	.long	.Linfo_string69                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	45                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x154:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc1                    # DW_AT_location
	.long	.Linfo_string70                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	45                              # DW_AT_decl_line
	.long	2002                            # DW_AT_type
	.byte	12                              # Abbrev [12] 0x163:0xf DW_TAG_variable
	.byte	3                               # DW_AT_location
	.byte	145
	.ascii	"\330\001"
	.long	.Linfo_string67                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
	.long	1343                            # DW_AT_type
	.byte	12                              # Abbrev [12] 0x172:0xf DW_TAG_variable
	.byte	3                               # DW_AT_location
	.byte	145
	.ascii	"\310\001"
	.long	.Linfo_string68                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
	.long	1343                            # DW_AT_type
	.byte	13                              # Abbrev [13] 0x181:0xf DW_TAG_variable
	.long	.Ldebug_loc2                    # DW_AT_location
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
	.long	2007                            # DW_AT_type
	.byte	13                              # Abbrev [13] 0x190:0xf DW_TAG_variable
	.long	.Ldebug_loc3                    # DW_AT_location
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	55                              # DW_AT_decl_line
	.long	2007                            # DW_AT_type
	.byte	13                              # Abbrev [13] 0x19f:0xf DW_TAG_variable
	.long	.Ldebug_loc4                    # DW_AT_location
	.long	.Linfo_string72                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	56                              # DW_AT_decl_line
	.long	2007                            # DW_AT_type
	.byte	13                              # Abbrev [13] 0x1ae:0xf DW_TAG_variable
	.long	.Ldebug_loc8                    # DW_AT_location
	.long	.Linfo_string77                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	120                             # DW_AT_decl_line
	.long	2030                            # DW_AT_type
	.byte	13                              # Abbrev [13] 0x1bd:0xf DW_TAG_variable
	.long	.Ldebug_loc9                    # DW_AT_location
	.long	.Linfo_string78                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	127                             # DW_AT_decl_line
	.long	1458                            # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1cc:0x32 DW_TAG_lexical_block
	.quad	.Ltmp11                         # DW_AT_low_pc
	.long	.Ltmp16-.Ltmp11                 # DW_AT_high_pc
	.byte	15                              # Abbrev [15] 0x1d9:0xb DW_TAG_variable
	.long	.Linfo_string79                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	65                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	14                              # Abbrev [14] 0x1e4:0x19 DW_TAG_lexical_block
	.quad	.Ltmp11                         # DW_AT_low_pc
	.long	.Ltmp16-.Ltmp11                 # DW_AT_high_pc
	.byte	15                              # Abbrev [15] 0x1f1:0xb DW_TAG_variable
	.long	.Linfo_string80                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	66                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0x1fe:0xe7 DW_TAG_lexical_block
	.long	.Ldebug_ranges0                 # DW_AT_ranges
	.byte	17                              # Abbrev [17] 0x203:0xc DW_TAG_variable
	.byte	0                               # DW_AT_const_value
	.long	.Linfo_string73                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	83                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	16                              # Abbrev [16] 0x20f:0xd5 DW_TAG_lexical_block
	.long	.Ldebug_ranges1                 # DW_AT_ranges
	.byte	15                              # Abbrev [15] 0x214:0xb DW_TAG_variable
	.long	.Linfo_string81                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	84                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	16                              # Abbrev [16] 0x21f:0xc4 DW_TAG_lexical_block
	.long	.Ldebug_ranges2                 # DW_AT_ranges
	.byte	15                              # Abbrev [15] 0x224:0xb DW_TAG_variable
	.long	.Linfo_string82                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	85                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	16                              # Abbrev [16] 0x22f:0xb3 DW_TAG_lexical_block
	.long	.Ldebug_ranges3                 # DW_AT_ranges
	.byte	13                              # Abbrev [13] 0x234:0xf DW_TAG_variable
	.long	.Ldebug_loc5                    # DW_AT_location
	.long	.Linfo_string74                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	88                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0x243:0xf DW_TAG_variable
	.long	.Ldebug_loc6                    # DW_AT_location
	.long	.Linfo_string75                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	89                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0x252:0xf DW_TAG_variable
	.long	.Ldebug_loc7                    # DW_AT_location
	.long	.Linfo_string76                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	90                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	16                              # Abbrev [16] 0x261:0x44 DW_TAG_lexical_block
	.long	.Ldebug_ranges4                 # DW_AT_ranges
	.byte	15                              # Abbrev [15] 0x266:0xb DW_TAG_variable
	.long	.Linfo_string79                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	95                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	16                              # Abbrev [16] 0x271:0x33 DW_TAG_lexical_block
	.long	.Ldebug_ranges5                 # DW_AT_ranges
	.byte	15                              # Abbrev [15] 0x276:0xb DW_TAG_variable
	.long	.Linfo_string83                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	96                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	16                              # Abbrev [16] 0x281:0x22 DW_TAG_lexical_block
	.long	.Ldebug_ranges6                 # DW_AT_ranges
	.byte	15                              # Abbrev [15] 0x286:0xb DW_TAG_variable
	.long	.Linfo_string84                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	99                              # DW_AT_decl_line
	.long	2030                            # DW_AT_type
	.byte	16                              # Abbrev [16] 0x291:0x11 DW_TAG_lexical_block
	.long	.Ldebug_ranges7                 # DW_AT_ranges
	.byte	15                              # Abbrev [15] 0x296:0xb DW_TAG_variable
	.long	.Linfo_string80                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	108                             # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0x2a5:0x14 DW_TAG_inlined_subroutine
	.long	258                             # DW_AT_abstract_origin
	.quad	.Ltmp27                         # DW_AT_low_pc
	.long	.Ltmp29-.Ltmp27                 # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	88                              # DW_AT_call_line
	.byte	29                              # DW_AT_call_column
	.byte	18                              # Abbrev [18] 0x2b9:0x14 DW_TAG_inlined_subroutine
	.long	258                             # DW_AT_abstract_origin
	.quad	.Ltmp36                         # DW_AT_low_pc
	.long	.Ltmp38-.Ltmp36                 # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	89                              # DW_AT_call_line
	.byte	29                              # DW_AT_call_column
	.byte	18                              # Abbrev [18] 0x2cd:0x14 DW_TAG_inlined_subroutine
	.long	258                             # DW_AT_abstract_origin
	.quad	.Ltmp50                         # DW_AT_low_pc
	.long	.Ltmp51-.Ltmp50                 # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	90                              # DW_AT_call_line
	.byte	29                              # DW_AT_call_column
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	14                              # Abbrev [14] 0x2e5:0x33 DW_TAG_lexical_block
	.quad	.Ltmp114                        # DW_AT_low_pc
	.long	.Ltmp128-.Ltmp114               # DW_AT_high_pc
	.byte	17                              # Abbrev [17] 0x2f2:0xc DW_TAG_variable
	.byte	0                               # DW_AT_const_value
	.long	.Linfo_string79                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	135                             # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	14                              # Abbrev [14] 0x2fe:0x19 DW_TAG_lexical_block
	.quad	.Ltmp114                        # DW_AT_low_pc
	.long	.Ltmp124-.Ltmp114               # DW_AT_high_pc
	.byte	15                              # Abbrev [15] 0x30b:0xb DW_TAG_variable
	.long	.Linfo_string80                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	137                             # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x318:0x1d DW_TAG_GNU_call_site
	.long	1251                            # DW_AT_abstract_origin
	.quad	.Ltmp3                          # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x325:0x9 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	5                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\200\204\257_"
	.byte	20                              # Abbrev [20] 0x32e:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	16
	.byte	64
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x335:0x1d DW_TAG_GNU_call_site
	.long	1251                            # DW_AT_abstract_origin
	.quad	.Ltmp5                          # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x342:0x9 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	5                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\200\204\257_"
	.byte	20                              # Abbrev [20] 0x34b:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	16
	.byte	64
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x352:0x1d DW_TAG_GNU_call_site
	.long	1251                            # DW_AT_abstract_origin
	.quad	.Ltmp7                          # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x35f:0x9 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	5                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\200\204\257_"
	.byte	20                              # Abbrev [20] 0x368:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	16
	.byte	64
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x36f:0x1a DW_TAG_GNU_call_site
	.long	1293                            # DW_AT_abstract_origin
	.quad	.Ltmp19                         # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x37c:0x5 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	1                               # DW_AT_GNU_call_site_value
	.byte	49
	.byte	20                              # Abbrev [20] 0x381:0x7 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	3                               # DW_AT_GNU_call_site_value
	.byte	145
	.ascii	"\330\001"
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x389:0xd DW_TAG_GNU_call_site
	.long	1405                            # DW_AT_abstract_origin
	.quad	.Ltmp103                        # DW_AT_low_pc
	.byte	19                              # Abbrev [19] 0x396:0x1a DW_TAG_GNU_call_site
	.long	1293                            # DW_AT_abstract_origin
	.quad	.Ltmp105                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x3a3:0x5 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	1                               # DW_AT_GNU_call_site_value
	.byte	49
	.byte	20                              # Abbrev [20] 0x3a8:0x7 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	3                               # DW_AT_GNU_call_site_value
	.byte	145
	.ascii	"\310\001"
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x3b0:0x1b DW_TAG_GNU_call_site
	.long	1429                            # DW_AT_abstract_origin
	.quad	.Ltmp107                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x3bd:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	82
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	16
	.byte	64
	.byte	20                              # Abbrev [20] 0x3c3:0x7 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	81
	.byte	3                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\210'"
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x3cb:0xd DW_TAG_GNU_call_site
	.long	1948                            # DW_AT_abstract_origin
	.quad	.Ltmp108                        # DW_AT_low_pc
	.byte	19                              # Abbrev [19] 0x3d8:0x15 DW_TAG_GNU_call_site
	.long	1429                            # DW_AT_abstract_origin
	.quad	.Ltmp112                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x3e5:0x7 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	81
	.byte	3                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\210'"
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x3ed:0x14 DW_TAG_GNU_call_site
	.long	1429                            # DW_AT_abstract_origin
	.quad	.Ltmp115                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x3fa:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x401:0x14 DW_TAG_GNU_call_site
	.long	1429                            # DW_AT_abstract_origin
	.quad	.Ltmp116                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x40e:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x415:0x14 DW_TAG_GNU_call_site
	.long	1429                            # DW_AT_abstract_origin
	.quad	.Ltmp117                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x422:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x429:0x14 DW_TAG_GNU_call_site
	.long	1429                            # DW_AT_abstract_origin
	.quad	.Ltmp118                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x436:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x43d:0x14 DW_TAG_GNU_call_site
	.long	1429                            # DW_AT_abstract_origin
	.quad	.Ltmp119                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x44a:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x451:0x14 DW_TAG_GNU_call_site
	.long	1429                            # DW_AT_abstract_origin
	.quad	.Ltmp120                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x45e:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x465:0x14 DW_TAG_GNU_call_site
	.long	1429                            # DW_AT_abstract_origin
	.quad	.Ltmp121                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x472:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x479:0x14 DW_TAG_GNU_call_site
	.long	1429                            # DW_AT_abstract_origin
	.quad	.Ltmp122                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x486:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x48d:0x14 DW_TAG_GNU_call_site
	.long	1971                            # DW_AT_abstract_origin
	.quad	.Ltmp129                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x49a:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x4a1:0x16 DW_TAG_GNU_call_site
	.long	1988                            # DW_AT_abstract_origin
	.quad	.Ltmp130                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x4ae:0x8 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	4                               # DW_AT_GNU_call_site_value
	.byte	145
	.byte	32
	.byte	148
	.byte	8
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x4b7:0x14 DW_TAG_GNU_call_site
	.long	1988                            # DW_AT_abstract_origin
	.quad	.Ltmp131                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x4c4:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	125
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x4cb:0x17 DW_TAG_GNU_call_site
	.long	1988                            # DW_AT_abstract_origin
	.quad	.Ltmp132                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x4d8:0x9 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	5                               # DW_AT_GNU_call_site_value
	.byte	145
	.asciz	"\310"
	.byte	148
	.byte	8
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	22                              # Abbrev [22] 0x4e3:0x17 DW_TAG_subprogram
	.long	.Linfo_string10                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	605                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1274                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	23                              # Abbrev [23] 0x4ef:0x5 DW_TAG_formal_parameter
	.long	1275                            # DW_AT_type
	.byte	23                              # Abbrev [23] 0x4f4:0x5 DW_TAG_formal_parameter
	.long	1275                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	24                              # Abbrev [24] 0x4fa:0x1 DW_TAG_pointer_type
	.byte	25                              # Abbrev [25] 0x4fb:0xb DW_TAG_typedef
	.long	1286                            # DW_AT_type
	.long	.Linfo_string12                 # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
	.byte	5                               # Abbrev [5] 0x506:0x7 DW_TAG_base_type
	.long	.Linfo_string11                 # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	22                              # Abbrev [22] 0x50d:0x17 DW_TAG_subprogram
	.long	.Linfo_string13                 # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.short	288                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	293                             # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	23                              # Abbrev [23] 0x519:0x5 DW_TAG_formal_parameter
	.long	1316                            # DW_AT_type
	.byte	23                              # Abbrev [23] 0x51e:0x5 DW_TAG_formal_parameter
	.long	1338                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	25                              # Abbrev [25] 0x524:0xb DW_TAG_typedef
	.long	1327                            # DW_AT_type
	.long	.Linfo_string15                 # DW_AT_name
	.byte	6                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0x52f:0xb DW_TAG_typedef
	.long	293                             # DW_AT_type
	.long	.Linfo_string14                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	169                             # DW_AT_decl_line
	.byte	26                              # Abbrev [26] 0x53a:0x5 DW_TAG_pointer_type
	.long	1343                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x53f:0x21 DW_TAG_structure_type
	.long	.Linfo_string21                 # DW_AT_name
	.byte	16                              # DW_AT_byte_size
	.byte	7                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.byte	28                              # Abbrev [28] 0x547:0xc DW_TAG_member
	.long	.Linfo_string16                 # DW_AT_name
	.long	1376                            # DW_AT_type
	.byte	7                               # DW_AT_decl_file
	.byte	16                              # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x553:0xc DW_TAG_member
	.long	.Linfo_string19                 # DW_AT_name
	.long	1394                            # DW_AT_type
	.byte	7                               # DW_AT_decl_file
	.byte	21                              # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	25                              # Abbrev [25] 0x560:0xb DW_TAG_typedef
	.long	1387                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	160                             # DW_AT_decl_line
	.byte	5                               # Abbrev [5] 0x56b:0x7 DW_TAG_base_type
	.long	.Linfo_string17                 # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	25                              # Abbrev [25] 0x572:0xb DW_TAG_typedef
	.long	1387                            # DW_AT_type
	.long	.Linfo_string20                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	197                             # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x57d:0xe DW_TAG_subprogram
	.long	.Linfo_string22                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.short	804                             # DW_AT_decl_line
                                        # DW_AT_prototyped
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	23                              # Abbrev [23] 0x585:0x5 DW_TAG_formal_parameter
	.long	1419                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	26                              # Abbrev [26] 0x58b:0x5 DW_TAG_pointer_type
	.long	1424                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x590:0x5 DW_TAG_const_type
	.long	75                              # DW_AT_type
	.byte	22                              # Abbrev [22] 0x595:0x18 DW_TAG_subprogram
	.long	.Linfo_string23                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.short	350                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	293                             # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	23                              # Abbrev [23] 0x5a1:0x5 DW_TAG_formal_parameter
	.long	1453                            # DW_AT_type
	.byte	23                              # Abbrev [23] 0x5a6:0x5 DW_TAG_formal_parameter
	.long	1943                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0x5ab:0x1 DW_TAG_unspecified_parameters
	.byte	0                               # End Of Children Mark
	.byte	32                              # Abbrev [32] 0x5ad:0x5 DW_TAG_restrict_type
	.long	1458                            # DW_AT_type
	.byte	26                              # Abbrev [26] 0x5b2:0x5 DW_TAG_pointer_type
	.long	1463                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x5b7:0xb DW_TAG_typedef
	.long	1474                            # DW_AT_type
	.long	.Linfo_string62                 # DW_AT_name
	.byte	10                              # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	27                              # Abbrev [27] 0x5c2:0x165 DW_TAG_structure_type
	.long	.Linfo_string61                 # DW_AT_name
	.byte	216                             # DW_AT_byte_size
	.byte	9                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
	.byte	28                              # Abbrev [28] 0x5ca:0xc DW_TAG_member
	.long	.Linfo_string24                 # DW_AT_name
	.long	293                             # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	51                              # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x5d6:0xc DW_TAG_member
	.long	.Linfo_string25                 # DW_AT_name
	.long	1831                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x5e2:0xc DW_TAG_member
	.long	.Linfo_string26                 # DW_AT_name
	.long	1831                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	55                              # DW_AT_decl_line
	.byte	16                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x5ee:0xc DW_TAG_member
	.long	.Linfo_string27                 # DW_AT_name
	.long	1831                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	56                              # DW_AT_decl_line
	.byte	24                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x5fa:0xc DW_TAG_member
	.long	.Linfo_string28                 # DW_AT_name
	.long	1831                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	57                              # DW_AT_decl_line
	.byte	32                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x606:0xc DW_TAG_member
	.long	.Linfo_string29                 # DW_AT_name
	.long	1831                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.byte	40                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x612:0xc DW_TAG_member
	.long	.Linfo_string30                 # DW_AT_name
	.long	1831                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
	.byte	48                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x61e:0xc DW_TAG_member
	.long	.Linfo_string31                 # DW_AT_name
	.long	1831                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	60                              # DW_AT_decl_line
	.byte	56                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x62a:0xc DW_TAG_member
	.long	.Linfo_string32                 # DW_AT_name
	.long	1831                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	61                              # DW_AT_decl_line
	.byte	64                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x636:0xc DW_TAG_member
	.long	.Linfo_string33                 # DW_AT_name
	.long	1831                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	64                              # DW_AT_decl_line
	.byte	72                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x642:0xc DW_TAG_member
	.long	.Linfo_string34                 # DW_AT_name
	.long	1831                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	65                              # DW_AT_decl_line
	.byte	80                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x64e:0xc DW_TAG_member
	.long	.Linfo_string35                 # DW_AT_name
	.long	1831                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	66                              # DW_AT_decl_line
	.byte	88                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x65a:0xc DW_TAG_member
	.long	.Linfo_string36                 # DW_AT_name
	.long	1836                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	68                              # DW_AT_decl_line
	.byte	96                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x666:0xc DW_TAG_member
	.long	.Linfo_string38                 # DW_AT_name
	.long	1846                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	70                              # DW_AT_decl_line
	.byte	104                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x672:0xc DW_TAG_member
	.long	.Linfo_string39                 # DW_AT_name
	.long	293                             # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	72                              # DW_AT_decl_line
	.byte	112                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x67e:0xc DW_TAG_member
	.long	.Linfo_string40                 # DW_AT_name
	.long	293                             # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.byte	116                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x68a:0xc DW_TAG_member
	.long	.Linfo_string41                 # DW_AT_name
	.long	1851                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	74                              # DW_AT_decl_line
	.byte	120                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x696:0xc DW_TAG_member
	.long	.Linfo_string43                 # DW_AT_name
	.long	1862                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	77                              # DW_AT_decl_line
	.byte	128                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6a2:0xc DW_TAG_member
	.long	.Linfo_string45                 # DW_AT_name
	.long	1869                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	78                              # DW_AT_decl_line
	.byte	130                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6ae:0xc DW_TAG_member
	.long	.Linfo_string47                 # DW_AT_name
	.long	1876                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	79                              # DW_AT_decl_line
	.byte	131                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6ba:0xc DW_TAG_member
	.long	.Linfo_string48                 # DW_AT_name
	.long	1888                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	81                              # DW_AT_decl_line
	.byte	136                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6c6:0xc DW_TAG_member
	.long	.Linfo_string50                 # DW_AT_name
	.long	1900                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	89                              # DW_AT_decl_line
	.byte	144                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6d2:0xc DW_TAG_member
	.long	.Linfo_string52                 # DW_AT_name
	.long	1911                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	91                              # DW_AT_decl_line
	.byte	152                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6de:0xc DW_TAG_member
	.long	.Linfo_string54                 # DW_AT_name
	.long	1921                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	92                              # DW_AT_decl_line
	.byte	160                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6ea:0xc DW_TAG_member
	.long	.Linfo_string56                 # DW_AT_name
	.long	1846                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	93                              # DW_AT_decl_line
	.byte	168                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6f6:0xc DW_TAG_member
	.long	.Linfo_string57                 # DW_AT_name
	.long	1274                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	94                              # DW_AT_decl_line
	.byte	176                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x702:0xc DW_TAG_member
	.long	.Linfo_string58                 # DW_AT_name
	.long	1275                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	95                              # DW_AT_decl_line
	.byte	184                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x70e:0xc DW_TAG_member
	.long	.Linfo_string59                 # DW_AT_name
	.long	293                             # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	96                              # DW_AT_decl_line
	.byte	192                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x71a:0xc DW_TAG_member
	.long	.Linfo_string60                 # DW_AT_name
	.long	1931                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	98                              # DW_AT_decl_line
	.byte	196                             # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	26                              # Abbrev [26] 0x727:0x5 DW_TAG_pointer_type
	.long	75                              # DW_AT_type
	.byte	26                              # Abbrev [26] 0x72c:0x5 DW_TAG_pointer_type
	.long	1841                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x731:0x5 DW_TAG_structure_type
	.long	.Linfo_string37                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	26                              # Abbrev [26] 0x736:0x5 DW_TAG_pointer_type
	.long	1474                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x73b:0xb DW_TAG_typedef
	.long	1387                            # DW_AT_type
	.long	.Linfo_string42                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	152                             # DW_AT_decl_line
	.byte	5                               # Abbrev [5] 0x746:0x7 DW_TAG_base_type
	.long	.Linfo_string44                 # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	2                               # DW_AT_byte_size
	.byte	5                               # Abbrev [5] 0x74d:0x7 DW_TAG_base_type
	.long	.Linfo_string46                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x754:0xc DW_TAG_array_type
	.long	75                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x759:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	26                              # Abbrev [26] 0x760:0x5 DW_TAG_pointer_type
	.long	1893                            # DW_AT_type
	.byte	34                              # Abbrev [34] 0x765:0x7 DW_TAG_typedef
	.long	.Linfo_string49                 # DW_AT_name
	.byte	9                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0x76c:0xb DW_TAG_typedef
	.long	1387                            # DW_AT_type
	.long	.Linfo_string51                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	153                             # DW_AT_decl_line
	.byte	26                              # Abbrev [26] 0x777:0x5 DW_TAG_pointer_type
	.long	1916                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x77c:0x5 DW_TAG_structure_type
	.long	.Linfo_string53                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	26                              # Abbrev [26] 0x781:0x5 DW_TAG_pointer_type
	.long	1926                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x786:0x5 DW_TAG_structure_type
	.long	.Linfo_string55                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	3                               # Abbrev [3] 0x78b:0xc DW_TAG_array_type
	.long	75                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x790:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	20                              # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	32                              # Abbrev [32] 0x797:0x5 DW_TAG_restrict_type
	.long	1419                            # DW_AT_type
	.byte	22                              # Abbrev [22] 0x79c:0x17 DW_TAG_subprogram
	.long	.Linfo_string63                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.short	258                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1458                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	23                              # Abbrev [23] 0x7a8:0x5 DW_TAG_formal_parameter
	.long	1943                            # DW_AT_type
	.byte	23                              # Abbrev [23] 0x7ad:0x5 DW_TAG_formal_parameter
	.long	1943                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	35                              # Abbrev [35] 0x7b3:0x11 DW_TAG_subprogram
	.long	.Linfo_string64                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	178                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	293                             # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	23                              # Abbrev [23] 0x7be:0x5 DW_TAG_formal_parameter
	.long	1458                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	29                              # Abbrev [29] 0x7c4:0xe DW_TAG_subprogram
	.long	.Linfo_string65                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	568                             # DW_AT_decl_line
                                        # DW_AT_prototyped
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	23                              # Abbrev [23] 0x7cc:0x5 DW_TAG_formal_parameter
	.long	1274                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	26                              # Abbrev [26] 0x7d2:0x5 DW_TAG_pointer_type
	.long	1831                            # DW_AT_type
	.byte	32                              # Abbrev [32] 0x7d7:0x5 DW_TAG_restrict_type
	.long	2012                            # DW_AT_type
	.byte	26                              # Abbrev [26] 0x7dc:0x5 DW_TAG_pointer_type
	.long	2017                            # DW_AT_type
	.byte	3                               # Abbrev [3] 0x7e1:0xd DW_TAG_array_type
	.long	2030                            # DW_AT_type
	.byte	36                              # Abbrev [36] 0x7e6:0x7 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.short	5000                            # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x7ee:0x7 DW_TAG_base_type
	.long	.Linfo_string71                 # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges5:
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges6:
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges7:
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang based Intel(R) oneAPI DPC++/C++ Compiler 2023.2.0 (2023.2.0.20230721)" # string offset=0
.Linfo_string1:
	.asciz	" --intel -g -O3 -x Host -funroll-loops -ffast-math -S -D ENABLE_TIMING src/matmul_seq_tile.c -o matmul_seq_tile_O3_xHost_funroll_ffastmath.s -fveclib=SVML -fheinous-gnu-extensions" # string offset=76
.Linfo_string2:
	.asciz	"src/matmul_seq_tile.c"         # string offset=256
.Linfo_string3:
	.asciz	"/home/Zsf/Desktop/_FINAL_PROJECT" # string offset=278
.Linfo_string4:
	.asciz	"char"                          # string offset=311
.Linfo_string5:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=316
.Linfo_string6:
	.asciz	"min"                           # string offset=336
.Linfo_string7:
	.asciz	"int"                           # string offset=340
.Linfo_string8:
	.asciz	"a"                             # string offset=344
.Linfo_string9:
	.asciz	"b"                             # string offset=346
.Linfo_string10:
	.asciz	"aligned_alloc"                 # string offset=348
.Linfo_string11:
	.asciz	"unsigned long"                 # string offset=362
.Linfo_string12:
	.asciz	"size_t"                        # string offset=376
.Linfo_string13:
	.asciz	"clock_gettime"                 # string offset=383
.Linfo_string14:
	.asciz	"__clockid_t"                   # string offset=397
.Linfo_string15:
	.asciz	"clockid_t"                     # string offset=409
.Linfo_string16:
	.asciz	"tv_sec"                        # string offset=419
.Linfo_string17:
	.asciz	"long"                          # string offset=426
.Linfo_string18:
	.asciz	"__time_t"                      # string offset=431
.Linfo_string19:
	.asciz	"tv_nsec"                       # string offset=440
.Linfo_string20:
	.asciz	"__syscall_slong_t"             # string offset=448
.Linfo_string21:
	.asciz	"timespec"                      # string offset=466
.Linfo_string22:
	.asciz	"perror"                        # string offset=475
.Linfo_string23:
	.asciz	"fprintf"                       # string offset=482
.Linfo_string24:
	.asciz	"_flags"                        # string offset=490
.Linfo_string25:
	.asciz	"_IO_read_ptr"                  # string offset=497
.Linfo_string26:
	.asciz	"_IO_read_end"                  # string offset=510
.Linfo_string27:
	.asciz	"_IO_read_base"                 # string offset=523
.Linfo_string28:
	.asciz	"_IO_write_base"                # string offset=537
.Linfo_string29:
	.asciz	"_IO_write_ptr"                 # string offset=552
.Linfo_string30:
	.asciz	"_IO_write_end"                 # string offset=566
.Linfo_string31:
	.asciz	"_IO_buf_base"                  # string offset=580
.Linfo_string32:
	.asciz	"_IO_buf_end"                   # string offset=593
.Linfo_string33:
	.asciz	"_IO_save_base"                 # string offset=605
.Linfo_string34:
	.asciz	"_IO_backup_base"               # string offset=619
.Linfo_string35:
	.asciz	"_IO_save_end"                  # string offset=635
.Linfo_string36:
	.asciz	"_markers"                      # string offset=648
.Linfo_string37:
	.asciz	"_IO_marker"                    # string offset=657
.Linfo_string38:
	.asciz	"_chain"                        # string offset=668
.Linfo_string39:
	.asciz	"_fileno"                       # string offset=675
.Linfo_string40:
	.asciz	"_flags2"                       # string offset=683
.Linfo_string41:
	.asciz	"_old_offset"                   # string offset=691
.Linfo_string42:
	.asciz	"__off_t"                       # string offset=703
.Linfo_string43:
	.asciz	"_cur_column"                   # string offset=711
.Linfo_string44:
	.asciz	"unsigned short"                # string offset=723
.Linfo_string45:
	.asciz	"_vtable_offset"                # string offset=738
.Linfo_string46:
	.asciz	"signed char"                   # string offset=753
.Linfo_string47:
	.asciz	"_shortbuf"                     # string offset=765
.Linfo_string48:
	.asciz	"_lock"                         # string offset=775
.Linfo_string49:
	.asciz	"_IO_lock_t"                    # string offset=781
.Linfo_string50:
	.asciz	"_offset"                       # string offset=792
.Linfo_string51:
	.asciz	"__off64_t"                     # string offset=800
.Linfo_string52:
	.asciz	"_codecvt"                      # string offset=810
.Linfo_string53:
	.asciz	"_IO_codecvt"                   # string offset=819
.Linfo_string54:
	.asciz	"_wide_data"                    # string offset=831
.Linfo_string55:
	.asciz	"_IO_wide_data"                 # string offset=842
.Linfo_string56:
	.asciz	"_freeres_list"                 # string offset=856
.Linfo_string57:
	.asciz	"_freeres_buf"                  # string offset=870
.Linfo_string58:
	.asciz	"__pad5"                        # string offset=883
.Linfo_string59:
	.asciz	"_mode"                         # string offset=890
.Linfo_string60:
	.asciz	"_unused2"                      # string offset=896
.Linfo_string61:
	.asciz	"_IO_FILE"                      # string offset=905
.Linfo_string62:
	.asciz	"FILE"                          # string offset=914
.Linfo_string63:
	.asciz	"fopen"                         # string offset=919
.Linfo_string64:
	.asciz	"fclose"                        # string offset=925
.Linfo_string65:
	.asciz	"free"                          # string offset=932
.Linfo_string66:
	.asciz	"main"                          # string offset=937
.Linfo_string67:
	.asciz	"start"                         # string offset=942
.Linfo_string68:
	.asciz	"end"                           # string offset=948
.Linfo_string69:
	.asciz	"argc"                          # string offset=952
.Linfo_string70:
	.asciz	"argv"                          # string offset=957
.Linfo_string71:
	.asciz	"double"                        # string offset=962
.Linfo_string72:
	.asciz	"c"                             # string offset=969
.Linfo_string73:
	.asciz	"ii"                            # string offset=971
.Linfo_string74:
	.asciz	"i_max"                         # string offset=974
.Linfo_string75:
	.asciz	"k_max"                         # string offset=980
.Linfo_string76:
	.asciz	"j_max"                         # string offset=986
.Linfo_string77:
	.asciz	"time_taken"                    # string offset=992
.Linfo_string78:
	.asciz	"f"                             # string offset=1003
.Linfo_string79:
	.asciz	"i"                             # string offset=1005
.Linfo_string80:
	.asciz	"j"                             # string offset=1007
.Linfo_string81:
	.asciz	"kk"                            # string offset=1009
.Linfo_string82:
	.asciz	"jj"                            # string offset=1012
.Linfo_string83:
	.asciz	"k"                             # string offset=1015
.Linfo_string84:
	.asciz	"r"                             # string offset=1017
	.ident	"Intel(R) oneAPI DPC++/C++ Compiler 2023.2.0 (2023.2.0.20230721)"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
