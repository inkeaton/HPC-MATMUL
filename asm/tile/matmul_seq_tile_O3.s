	.text
	.file	"matmul_seq_tile.c"
	.file	1 "/home/Zsf/Desktop/_FINAL_PROJECT" "src/matmul_seq_tile.c"
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
	subq	$168, %rsp
	.cfi_def_cfa_offset 224
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	stmxcsr	72(%rsp)
	orl	$32832, 72(%rsp)                # imm = 0x8040
	ldmxcsr	72(%rsp)
.Ltmp0:
	.loc	1 54 32 prologue_end            # src/matmul_seq_tile.c:54:32
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
	.loc	1 55 32                         # src/matmul_seq_tile.c:55:32
	movl	$64, %edi
	movl	$200000000, %esi                # imm = 0xBEBC200
	callq	aligned_alloc
.Ltmp5:
	movq	%rax, %r14
.Ltmp6:
	#DEBUG_VALUE: main:b <- $r14
	.loc	1 56 32                         # src/matmul_seq_tile.c:56:32
	movl	$64, %edi
	movl	$200000000, %esi                # imm = 0xBEBC200
	callq	aligned_alloc
.Ltmp7:
	#DEBUG_VALUE: main:c <- $rax
	.loc	1 58 10                         # src/matmul_seq_tile.c:58:10
	testq	%rbx, %rbx
	.loc	1 58 12 is_stmt 0               # src/matmul_seq_tile.c:58:12
	je	.LBB0_16
.Ltmp8:
# %bb.1:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $rax
	testq	%r14, %r14
	je	.LBB0_16
.Ltmp9:
# %bb.2:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $rax
	.loc	1 56 32 is_stmt 1               # src/matmul_seq_tile.c:56:32
	movq	%rax, %r15
.Ltmp10:
	.loc	1 58 12                         # src/matmul_seq_tile.c:58:12
	testq	%rax, %rax
	je	.LBB0_16
.Ltmp11:
# %bb.3:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r15
	.loc	1 70 21                         # src/matmul_seq_tile.c:70:21
	movl	$200000000, %edx                # imm = 0xBEBC200
	movq	%r15, %rdi
	xorl	%esi, %esi
	callq	_intel_fast_memset@PLT
.Ltmp12:
	.loc	1 0 21 is_stmt 0                # src/matmul_seq_tile.c:0:21
	movl	$48, %eax
	.loc	1 68 21 is_stmt 1 discriminator 2 # src/matmul_seq_tile.c:68:21
	movapd	.LCPI0_0(%rip), %xmm0           # xmm0 = [2.0E+0,2.0E+0]
	.loc	1 69 21 discriminator 2         # src/matmul_seq_tile.c:69:21
	movapd	.LCPI0_1(%rip), %xmm1           # xmm1 = [3.0E+0,3.0E+0]
.Ltmp13:
	.p2align	4, 0x90
.LBB0_4:                                # =>This Inner Loop Header: Depth=1
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r15
	.loc	1 68 21                         # src/matmul_seq_tile.c:68:21
	movapd	%xmm0, -48(%rbx,%rax)
	.loc	1 69 21                         # src/matmul_seq_tile.c:69:21
	movapd	%xmm1, -48(%r14,%rax)
	.loc	1 68 21                         # src/matmul_seq_tile.c:68:21
	movapd	%xmm0, -32(%rbx,%rax)
	.loc	1 69 21                         # src/matmul_seq_tile.c:69:21
	movapd	%xmm1, -32(%r14,%rax)
	.loc	1 68 21                         # src/matmul_seq_tile.c:68:21
	movapd	%xmm0, -16(%rbx,%rax)
	.loc	1 69 21                         # src/matmul_seq_tile.c:69:21
	movapd	%xmm1, -16(%r14,%rax)
	.loc	1 68 21                         # src/matmul_seq_tile.c:68:21
	movapd	%xmm0, (%rbx,%rax)
	.loc	1 69 21                         # src/matmul_seq_tile.c:69:21
	movapd	%xmm1, (%r14,%rax)
.Ltmp14:
	.loc	1 66 27                         # src/matmul_seq_tile.c:66:27
	addq	$64, %rax
	cmpq	$200000048, %rax                # imm = 0xBEBC230
.Ltmp15:
	.loc	1 66 9 is_stmt 0                # src/matmul_seq_tile.c:66:9
	jne	.LBB0_4
.Ltmp16:
# %bb.5:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $r15
	.loc	1 0 9                           # src/matmul_seq_tile.c:0:9
	leaq	72(%rsp), %rsi
	.loc	1 76 9 is_stmt 1                # src/matmul_seq_tile.c:76:9
	movl	$1, %edi
	callq	clock_gettime
.Ltmp17:
	.loc	1 0 9 is_stmt 0                 # src/matmul_seq_tile.c:0:9
	xorl	%eax, %eax
	#DEBUG_VALUE: ii <- 0
	movq	%r15, 88(%rsp)                  # 8-byte Spill
.Ltmp18:
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	movq	%r15, 40(%rsp)                  # 8-byte Spill
	movq	%r14, 32(%rsp)                  # 8-byte Spill
.Ltmp19:
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	jmp	.LBB0_6
.Ltmp20:
	.p2align	4, 0x90
.LBB0_24:                               #   in Loop: Header=BB0_6 Depth=1
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	movq	96(%rsp), %rcx                  # 8-byte Reload
.Ltmp21:
	.loc	1 83 25 is_stmt 1               # src/matmul_seq_tile.c:83:25
	leal	1(%rcx), %eax
.Ltmp22:
	.loc	1 83 5 is_stmt 0                # src/matmul_seq_tile.c:83:5
	addq	$2560000, 40(%rsp)              # 8-byte Folded Spill
                                        # imm = 0x271000
.Ltmp23:
	.loc	1 83 25                         # src/matmul_seq_tile.c:83:25
	cmpl	$78, %ecx
                                        # kill: def $eax killed $eax def $rax
	movq	32(%rsp), %r14                  # 8-byte Reload
.Ltmp24:
	.loc	1 83 5                          # src/matmul_seq_tile.c:83:5
	je	.LBB0_25
.Ltmp25:
.LBB0_6:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_7 Depth 2
                                        #       Child Loop BB0_8 Depth 3
                                        #         Child Loop BB0_10 Depth 4
                                        #           Child Loop BB0_12 Depth 5
                                        #             Child Loop BB0_19 Depth 6
                                        #             Child Loop BB0_15 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	.loc	1 43 39 is_stmt 1               # src/matmul_seq_tile.c:43:39
	movl	%eax, %ecx
	shll	$6, %ecx
	cmpl	$4936, %ecx                     # imm = 0x1348
	movl	$4936, %edx                     # imm = 0x1348
	movl	%ecx, 24(%rsp)                  # 4-byte Spill
	cmovll	%ecx, %edx
.Ltmp26:
	.loc	1 95 36                         # src/matmul_seq_tile.c:95:36
	leal	64(%rdx), %ecx
	movl	%ecx, 20(%rsp)                  # 4-byte Spill
	movq	%rax, 96(%rsp)                  # 8-byte Spill
.Ltmp27:
	.loc	1 95 17 is_stmt 0               # src/matmul_seq_tile.c:95:17
	movl	%eax, %eax
	shlq	$6, %rax
	movq	%rax, 120(%rsp)                 # 8-byte Spill
	subq	%rax, %rdx
	addq	$63, %rdx
	movq	%rdx, 128(%rsp)                 # 8-byte Spill
	xorl	%ecx, %ecx
	jmp	.LBB0_7
.Ltmp28:
	.p2align	4, 0x90
.LBB0_23:                               #   in Loop: Header=BB0_7 Depth=2
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	.loc	1 0 17                          # src/matmul_seq_tile.c:0:17
	movq	104(%rsp), %rcx                 # 8-byte Reload
	.loc	1 84 29 is_stmt 1               # src/matmul_seq_tile.c:84:29
	leal	1(%rcx), %eax
	movq	112(%rsp), %r14                 # 8-byte Reload
.Ltmp29:
	.loc	1 84 9 is_stmt 0                # src/matmul_seq_tile.c:84:9
	addq	$2560000, %r14                  # imm = 0x271000
.Ltmp30:
	.loc	1 84 29                         # src/matmul_seq_tile.c:84:29
	cmpl	$78, %ecx
	movl	%eax, %ecx
.Ltmp31:
	.loc	1 84 9                          # src/matmul_seq_tile.c:84:9
	je	.LBB0_24
.Ltmp32:
.LBB0_7:                                #   Parent Loop BB0_6 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_8 Depth 3
                                        #         Child Loop BB0_10 Depth 4
                                        #           Child Loop BB0_12 Depth 5
                                        #             Child Loop BB0_19 Depth 6
                                        #             Child Loop BB0_15 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	.loc	1 43 39 is_stmt 1               # src/matmul_seq_tile.c:43:39
	movl	%ecx, %eax
	shll	$6, %eax
	movq	%rax, 144(%rsp)                 # 8-byte Spill
	leal	64(%rax), %edx
	cmpl	$5000, %edx                     # imm = 0x1388
	movl	$5000, %eax                     # imm = 0x1388
	cmovgel	%eax, %edx
.Ltmp33:
	#DEBUG_VALUE: k_max <- $edx
	.loc	1 0 39 is_stmt 0                # src/matmul_seq_tile.c:0:39
	movq	%rcx, 104(%rsp)                 # 8-byte Spill
.Ltmp34:
	.loc	1 96 21 is_stmt 1               # src/matmul_seq_tile.c:96:21
	movl	%ecx, %r15d
	shlq	$6, %r15
	movq	%r15, %rbp
	notq	%rbp
	movq	%rdx, 136(%rsp)                 # 8-byte Spill
.Ltmp35:
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 136, DW_OP_deref] $rsp
	addq	%rdx, %rbp
	xorl	%eax, %eax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	movl	$64, %ecx
	movq	%r14, 112(%rsp)                 # 8-byte Spill
	movq	%r14, 64(%rsp)                  # 8-byte Spill
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	xorl	%eax, %eax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	jmp	.LBB0_8
.Ltmp36:
	.p2align	4, 0x90
.LBB0_22:                               #   in Loop: Header=BB0_8 Depth=3
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	.loc	1 0 21 is_stmt 0                # src/matmul_seq_tile.c:0:21
	movq	8(%rsp), %rdx                   # 8-byte Reload
	.loc	1 85 33 is_stmt 1               # src/matmul_seq_tile.c:85:33
	leal	1(%rdx), %eax
.Ltmp37:
	.loc	1 85 13 is_stmt 0               # src/matmul_seq_tile.c:85:13
	addq	$512, 48(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x200
	addq	$512, 64(%rsp)                  # 8-byte Folded Spill
                                        # imm = 0x200
	addq	$-64, 56(%rsp)                  # 8-byte Folded Spill
	movl	28(%rsp), %ecx                  # 4-byte Reload
	addl	$64, %ecx
.Ltmp38:
	.loc	1 85 33                         # src/matmul_seq_tile.c:85:33
	cmpl	$78, %edx
                                        # kill: def $eax killed $eax def $rax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
.Ltmp39:
	.loc	1 85 13                         # src/matmul_seq_tile.c:85:13
	je	.LBB0_23
.Ltmp40:
.LBB0_8:                                #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_10 Depth 4
                                        #           Child Loop BB0_12 Depth 5
                                        #             Child Loop BB0_19 Depth 6
                                        #             Child Loop BB0_15 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 136, DW_OP_deref] $rsp
	.loc	1 95 17 is_stmt 1               # src/matmul_seq_tile.c:95:17
	cmpl	$5000, %ecx                     # imm = 0x1388
	movl	$5000, %eax                     # imm = 0x1388
	movl	%ecx, 28(%rsp)                  # 4-byte Spill
	cmovll	%ecx, %eax
	movl	20(%rsp), %ecx                  # 4-byte Reload
.Ltmp41:
	.loc	1 95 36 is_stmt 0               # src/matmul_seq_tile.c:95:36
	cmpl	%ecx, 24(%rsp)                  # 4-byte Folded Reload
.Ltmp42:
	.loc	1 95 17                         # src/matmul_seq_tile.c:95:17
	jae	.LBB0_22
.Ltmp43:
# %bb.9:                                #   in Loop: Header=BB0_8 Depth=3
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 136, DW_OP_deref] $rsp
	addq	56(%rsp), %rax                  # 8-byte Folded Reload
	movq	8(%rsp), %rdx                   # 8-byte Reload
.Ltmp44:
	.loc	1 43 39 is_stmt 1               # src/matmul_seq_tile.c:43:39
	movl	%edx, %r11d
	shll	$6, %r11d
	leal	64(%r11), %r14d
	cmpl	$5000, %r14d                    # imm = 0x1388
	movl	$5000, %ecx                     # imm = 0x1388
	cmovgel	%ecx, %r14d
.Ltmp45:
	#DEBUG_VALUE: j_max <- $r14d
	.loc	1 108 44                        # src/matmul_seq_tile.c:108:44
	movl	%edx, %r9d
	shlq	$6, %r9
	movq	%r14, %r8
	subq	%r9, %r8
	movq	48(%rsp), %rdx                  # 8-byte Reload
	xorl	%esi, %esi
	jmp	.LBB0_10
.Ltmp46:
	.p2align	4, 0x90
.LBB0_21:                               #   in Loop: Header=BB0_10 Depth=4
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 136, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r14d
	.loc	1 95 36                         # src/matmul_seq_tile.c:95:36
	leaq	1(%rsi), %rcx
.Ltmp47:
	.loc	1 95 17 is_stmt 0               # src/matmul_seq_tile.c:95:17
	addq	$40000, %rdx                    # imm = 0x9C40
.Ltmp48:
	.loc	1 95 36                         # src/matmul_seq_tile.c:95:36
	cmpq	128(%rsp), %rsi                 # 8-byte Folded Reload
	movq	%rcx, %rsi
.Ltmp49:
	.loc	1 95 17                         # src/matmul_seq_tile.c:95:17
	je	.LBB0_22
.Ltmp50:
.LBB0_10:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_8 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_12 Depth 5
                                        #             Child Loop BB0_19 Depth 6
                                        #             Child Loop BB0_15 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 136, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r14d
	.loc	1 0 17                          # src/matmul_seq_tile.c:0:17
	movq	136(%rsp), %rcx                 # 8-byte Reload
.Ltmp51:
	.loc	1 96 40 is_stmt 1               # src/matmul_seq_tile.c:96:40
	cmpl	%ecx, 144(%rsp)                 # 4-byte Folded Reload
.Ltmp52:
	.loc	1 96 21 is_stmt 0               # src/matmul_seq_tile.c:96:21
	jae	.LBB0_21
.Ltmp53:
# %bb.11:                               #   in Loop: Header=BB0_10 Depth=4
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 136, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r14d
	.loc	1 0 21                          # src/matmul_seq_tile.c:0:21
	movq	120(%rsp), %rcx                 # 8-byte Reload
.Ltmp54:
	.loc	1 99 36 is_stmt 1               # src/matmul_seq_tile.c:99:36
	addq	%rsi, %rcx
	movq	64(%rsp), %r10                  # 8-byte Reload
	xorl	%edi, %edi
	jmp	.LBB0_12
.Ltmp55:
	.p2align	4, 0x90
.LBB0_20:                               #   in Loop: Header=BB0_12 Depth=5
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 136, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r14d
	.loc	1 96 40                         # src/matmul_seq_tile.c:96:40
	leaq	1(%rdi), %r12
.Ltmp56:
	.loc	1 96 21 is_stmt 0               # src/matmul_seq_tile.c:96:21
	addq	$40000, %r10                    # imm = 0x9C40
.Ltmp57:
	.loc	1 96 40                         # src/matmul_seq_tile.c:96:40
	cmpq	%rbp, %rdi
	movq	%r12, %rdi
.Ltmp58:
	.loc	1 96 21                         # src/matmul_seq_tile.c:96:21
	je	.LBB0_21
.Ltmp59:
.LBB0_12:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_8 Depth=3
                                        #         Parent Loop BB0_10 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB0_19 Depth 6
                                        #             Child Loop BB0_15 Depth 6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 136, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r14d
	.loc	1 108 44 is_stmt 1              # src/matmul_seq_tile.c:108:44
	cmpl	%r14d, %r11d
.Ltmp60:
	#DEBUG_VALUE: r <- undef
	.loc	1 108 25 is_stmt 0              # src/matmul_seq_tile.c:108:25
	jae	.LBB0_20
.Ltmp61:
# %bb.13:                               #   in Loop: Header=BB0_12 Depth=5
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 136, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r14d
	.loc	1 99 36 is_stmt 1               # src/matmul_seq_tile.c:99:36
	leaq	(%r15,%rdi), %r12
	imulq	$40000, %rcx, %r13              # imm = 0x9C40
	addq	%rbx, %r13
	movsd	(%r13,%r12,8), %xmm0            # xmm0 = mem[0],zero
.Ltmp62:
	#DEBUG_VALUE: r <- $xmm0
	.loc	1 58 12                         # src/matmul_seq_tile.c:58:12
	cmpq	%r14, %r9
.Ltmp63:
	.loc	1 108 25                        # src/matmul_seq_tile.c:108:25
	jne	.LBB0_18
.Ltmp64:
# %bb.14:                               #   in Loop: Header=BB0_12 Depth=5
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 136, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r14d
	#DEBUG_VALUE: r <- $xmm0
	.loc	1 0 25 is_stmt 0                # src/matmul_seq_tile.c:0:25
	xorl	%r12d, %r12d
.Ltmp65:
	.p2align	4, 0x90
.LBB0_15:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_8 Depth=3
                                        #         Parent Loop BB0_10 Depth=4
                                        #           Parent Loop BB0_12 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 136, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r14d
	#DEBUG_VALUE: r <- $xmm0
	movsd	(%r10,%r12,8), %xmm1            # xmm1 = mem[0],zero
.Ltmp66:
	.loc	1 109 42 is_stmt 1              # src/matmul_seq_tile.c:109:42
	mulsd	%xmm0, %xmm1
	.loc	1 109 37 is_stmt 0              # src/matmul_seq_tile.c:109:37
	addsd	(%rdx,%r12,8), %xmm1
	movsd	%xmm1, (%rdx,%r12,8)
.Ltmp67:
	.loc	1 108 44 is_stmt 1              # src/matmul_seq_tile.c:108:44
	incq	%r12
	cmpq	%r12, %rax
	jne	.LBB0_15
	jmp	.LBB0_20
.Ltmp68:
	.p2align	4, 0x90
.LBB0_18:                               #   in Loop: Header=BB0_12 Depth=5
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 136, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r14d
	#DEBUG_VALUE: r <- $xmm0
	.loc	1 109 42                        # src/matmul_seq_tile.c:109:42
	unpcklpd	%xmm0, %xmm0                    # xmm0 = xmm0[0,0]
.Ltmp69:
	.loc	1 0 42 is_stmt 0                # src/matmul_seq_tile.c:0:42
	xorl	%r12d, %r12d
.Ltmp70:
	.p2align	4, 0x90
.LBB0_19:                               #   Parent Loop BB0_6 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_8 Depth=3
                                        #         Parent Loop BB0_10 Depth=4
                                        #           Parent Loop BB0_12 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: ii <- 0
	#DEBUG_VALUE: k_max <- [DW_OP_plus_uconst 136, DW_OP_deref] $rsp
	#DEBUG_VALUE: j_max <- $r14d
	movapd	(%r10,%r12,8), %xmm1
	.loc	1 109 42                        # src/matmul_seq_tile.c:109:42
	mulpd	%xmm0, %xmm1
	.loc	1 109 37                        # src/matmul_seq_tile.c:109:37
	addpd	(%rdx,%r12,8), %xmm1
	movapd	%xmm1, (%rdx,%r12,8)
.Ltmp71:
	.loc	1 108 44 is_stmt 1              # src/matmul_seq_tile.c:108:44
	addq	$2, %r12
	cmpq	%r8, %r12
.Ltmp72:
	.loc	1 108 25 is_stmt 0              # src/matmul_seq_tile.c:108:25
	jl	.LBB0_19
	jmp	.LBB0_20
.Ltmp73:
.LBB0_16:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- $r14
	#DEBUG_VALUE: main:c <- $rax
	.loc	1 59 9 is_stmt 1                # src/matmul_seq_tile.c:59:9
	movl	$.L.str, %edi
	callq	perror
.Ltmp74:
	.loc	1 0 9 is_stmt 0                 # src/matmul_seq_tile.c:0:9
	movl	$1, %ebp
	jmp	.LBB0_17
.Ltmp75:
.LBB0_25:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	movl	$1, %ebp
	leaq	152(%rsp), %rsi
	.loc	1 119 9 is_stmt 1               # src/matmul_seq_tile.c:119:9
	movl	$1, %edi
	callq	clock_gettime
.Ltmp76:
	.loc	1 120 34                        # src/matmul_seq_tile.c:120:34
	movq	152(%rsp), %rax
	.loc	1 120 64 is_stmt 0              # src/matmul_seq_tile.c:120:64
	movq	160(%rsp), %rcx
	.loc	1 120 41                        # src/matmul_seq_tile.c:120:41
	subq	72(%rsp), %rax
	.loc	1 120 29                        # src/matmul_seq_tile.c:120:29
	xorps	%xmm1, %xmm1
	cvtsi2sd	%rax, %xmm1
	.loc	1 120 72                        # src/matmul_seq_tile.c:120:72
	subq	80(%rsp), %rcx
	.loc	1 120 59                        # src/matmul_seq_tile.c:120:59
	xorps	%xmm0, %xmm0
	cvtsi2sd	%rcx, %xmm0
	.loc	1 120 89                        # src/matmul_seq_tile.c:120:89
	mulsd	.LCPI0_2(%rip), %xmm0
	.loc	1 120 57                        # src/matmul_seq_tile.c:120:57
	addsd	%xmm1, %xmm0
.Ltmp77:
	#DEBUG_VALUE: main:time_taken <- $xmm0
	.loc	1 122 17 is_stmt 1              # src/matmul_seq_tile.c:122:17
	movq	stderr(%rip), %rdi
	.loc	1 122 9 is_stmt 0               # src/matmul_seq_tile.c:122:9
	movl	$.L.str.1, %esi
	movl	$5000, %edx                     # imm = 0x1388
	movl	$64, %ecx
	movb	$1, %al
	callq	fprintf
.Ltmp78:
	.loc	1 127 15 is_stmt 1              # src/matmul_seq_tile.c:127:15
	movl	$.L.str.2, %edi
	movl	$.L.str.3, %esi
	callq	fopen
.Ltmp79:
	#DEBUG_VALUE: main:f <- $rax
	.loc	1 128 10                        # src/matmul_seq_tile.c:128:10
	testq	%rax, %rax
.Ltmp80:
	.loc	1 128 9 is_stmt 0               # src/matmul_seq_tile.c:128:9
	je	.LBB0_26
.Ltmp81:
# %bb.27:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $rax
	.loc	1 127 15 is_stmt 1              # src/matmul_seq_tile.c:127:15
	movq	%rax, %r12
	xorl	%r14d, %r14d
	.loc	1 134 5                         # src/matmul_seq_tile.c:134:5
	movl	$.L.str.5, %esi
	movq	%rax, %rdi
	movl	$5000, %edx                     # imm = 0x1388
	xorl	%eax, %eax
.Ltmp82:
	#DEBUG_VALUE: main:f <- $r12
	callq	fprintf
.Ltmp83:
	#DEBUG_VALUE: i <- 0
	.loc	1 0 5 is_stmt 0                 # src/matmul_seq_tile.c:0:5
	movq	88(%rsp), %rbp                  # 8-byte Reload
	movq	%rbp, %r15
	addq	$56, %r15
.Ltmp84:
	.p2align	4, 0x90
.LBB0_28:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_29 Depth 2
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	#DEBUG_VALUE: i <- 0
	xorl	%r13d, %r13d
.Ltmp85:
	.p2align	4, 0x90
.LBB0_29:                               #   Parent Loop BB0_28 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	#DEBUG_VALUE: i <- 0
	.loc	1 139 33 is_stmt 1              # src/matmul_seq_tile.c:139:33
	movsd	-56(%r15,%r13), %xmm0           # xmm0 = mem[0],zero
	.loc	1 139 13 is_stmt 0              # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp86:
	.loc	1 139 33                        # src/matmul_seq_tile.c:139:33
	movsd	-48(%r15,%r13), %xmm0           # xmm0 = mem[0],zero
	.loc	1 139 13                        # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp87:
	.loc	1 139 33                        # src/matmul_seq_tile.c:139:33
	movsd	-40(%r15,%r13), %xmm0           # xmm0 = mem[0],zero
	.loc	1 139 13                        # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp88:
	.loc	1 139 33                        # src/matmul_seq_tile.c:139:33
	movsd	-32(%r15,%r13), %xmm0           # xmm0 = mem[0],zero
	.loc	1 139 13                        # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp89:
	.loc	1 139 33                        # src/matmul_seq_tile.c:139:33
	movsd	-24(%r15,%r13), %xmm0           # xmm0 = mem[0],zero
	.loc	1 139 13                        # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp90:
	.loc	1 139 33                        # src/matmul_seq_tile.c:139:33
	movsd	-16(%r15,%r13), %xmm0           # xmm0 = mem[0],zero
	.loc	1 139 13                        # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp91:
	.loc	1 139 33                        # src/matmul_seq_tile.c:139:33
	movsd	-8(%r15,%r13), %xmm0            # xmm0 = mem[0],zero
	.loc	1 139 13                        # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp92:
	.loc	1 139 33                        # src/matmul_seq_tile.c:139:33
	movsd	(%r15,%r13), %xmm0              # xmm0 = mem[0],zero
	.loc	1 139 13                        # src/matmul_seq_tile.c:139:13
	movl	$.L.str.6, %esi
	movq	%r12, %rdi
	movb	$1, %al
	callq	fprintf
.Ltmp93:
	.loc	1 137 27 is_stmt 1              # src/matmul_seq_tile.c:137:27
	addq	$64, %r13
	cmpq	$8000, %r13                     # imm = 0x1F40
.Ltmp94:
	.loc	1 137 9 is_stmt 0               # src/matmul_seq_tile.c:137:9
	jne	.LBB0_29
.Ltmp95:
# %bb.30:                               #   in Loop: Header=BB0_28 Depth=1
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	#DEBUG_VALUE: i <- 0
	.loc	1 141 9 is_stmt 1               # src/matmul_seq_tile.c:141:9
	movl	$10, %edi
	movq	%r12, %rsi
	callq	fputc@PLT
.Ltmp96:
	.loc	1 135 23                        # src/matmul_seq_tile.c:135:23
	leaq	1(%r14), %rax
.Ltmp97:
	.loc	1 135 5 is_stmt 0               # src/matmul_seq_tile.c:135:5
	addq	$40000, %r15                    # imm = 0x9C40
.Ltmp98:
	.loc	1 135 23                        # src/matmul_seq_tile.c:135:23
	cmpq	$999, %r14                      # imm = 0x3E7
	movq	%rax, %r14
.Ltmp99:
	.loc	1 135 5                         # src/matmul_seq_tile.c:135:5
	jne	.LBB0_28
.Ltmp100:
# %bb.31:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $r12
	.loc	1 144 5 is_stmt 1               # src/matmul_seq_tile.c:144:5
	movq	%r12, %rdi
	callq	fclose
.Ltmp101:
	.loc	1 147 5                         # src/matmul_seq_tile.c:147:5
	movq	%rbx, %rdi
	callq	free
.Ltmp102:
	.loc	1 0 5 is_stmt 0                 # src/matmul_seq_tile.c:0:5
	movq	32(%rsp), %rdi                  # 8-byte Reload
	.loc	1 148 5 is_stmt 1               # src/matmul_seq_tile.c:148:5
	callq	free
.Ltmp103:
	.loc	1 149 5                         # src/matmul_seq_tile.c:149:5
	movq	%rbp, %rdi
	callq	free
.Ltmp104:
	.loc	1 0 5 is_stmt 0                 # src/matmul_seq_tile.c:0:5
	xorl	%ebp, %ebp
.Ltmp105:
.LBB0_17:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	.loc	1 151 1 is_stmt 1               # src/matmul_seq_tile.c:151:1
	movl	%ebp, %eax
	.loc	1 151 1 epilogue_begin is_stmt 0 # src/matmul_seq_tile.c:151:1
	addq	$168, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
.Ltmp106:
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Ltmp107:
.LBB0_26:
	.cfi_def_cfa_offset 224
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:a <- $rbx
	#DEBUG_VALUE: main:b <- [DW_OP_plus_uconst 32, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:c <- [DW_OP_plus_uconst 88, DW_OP_deref] $rsp
	#DEBUG_VALUE: main:f <- $rax
	.loc	1 130 9 is_stmt 1               # src/matmul_seq_tile.c:130:9
	movl	$.L.str.4, %edi
	callq	perror
.Ltmp108:
	.loc	1 0 9 is_stmt 0                 # src/matmul_seq_tile.c:0:9
	jmp	.LBB0_17
.Ltmp109:
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
	.quad	.Ltmp106-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	83                              # DW_OP_reg3
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	83                              # DW_OP_reg3
	.quad	0
	.quad	0
.Ldebug_loc3:
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	94                              # DW_OP_reg14
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.short	2                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	32                              # 32
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	94                              # DW_OP_reg14
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.short	2                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	32                              # 32
	.quad	.Ltmp107-.Lfunc_begin0
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
	.quad	.Ltmp18-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	95                              # DW_OP_reg15
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	216                             # 88
	.byte	0                               # 
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	80                              # DW_OP_reg0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	216                             # 88
	.byte	0                               # 
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	216                             # 88
	.byte	0                               # 
	.quad	0
	.quad	0
.Ldebug_loc5:
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	81                              # super-register DW_OP_reg1
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	136                             # 136
	.byte	1                               # 
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.short	3                               # Loc expr size
	.byte	119                             # DW_OP_breg7
	.byte	136                             # 136
	.byte	1                               # 
	.quad	0
	.quad	0
.Ldebug_loc6:
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	94                              # super-register DW_OP_reg14
	.quad	0
	.quad	0
.Ldebug_loc7:
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	97                              # DW_OP_reg17
	.quad	0
	.quad	0
.Ldebug_loc8:
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	97                              # DW_OP_reg17
	.quad	0
	.quad	0
.Ldebug_loc9:
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	80                              # DW_OP_reg0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	92                              # DW_OP_reg12
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
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
	.byte	1                               # Abbrev [1] 0xb:0x7f5 DW_TAG_compile_unit
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
	.byte	10                              # Abbrev [10] 0x12c:0x3c1 DW_TAG_subprogram
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
	.long	2012                            # DW_AT_type
	.byte	12                              # Abbrev [12] 0x163:0xf DW_TAG_variable
	.byte	3                               # DW_AT_location
	.byte	145
	.asciz	"\310"
	.long	.Linfo_string67                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
	.long	1353                            # DW_AT_type
	.byte	12                              # Abbrev [12] 0x172:0xf DW_TAG_variable
	.byte	3                               # DW_AT_location
	.byte	145
	.ascii	"\230\001"
	.long	.Linfo_string68                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	75                              # DW_AT_decl_line
	.long	1353                            # DW_AT_type
	.byte	13                              # Abbrev [13] 0x181:0xf DW_TAG_variable
	.long	.Ldebug_loc2                    # DW_AT_location
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
	.long	2017                            # DW_AT_type
	.byte	13                              # Abbrev [13] 0x190:0xf DW_TAG_variable
	.long	.Ldebug_loc3                    # DW_AT_location
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	55                              # DW_AT_decl_line
	.long	2017                            # DW_AT_type
	.byte	13                              # Abbrev [13] 0x19f:0xf DW_TAG_variable
	.long	.Ldebug_loc4                    # DW_AT_location
	.long	.Linfo_string72                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	56                              # DW_AT_decl_line
	.long	2017                            # DW_AT_type
	.byte	13                              # Abbrev [13] 0x1ae:0xf DW_TAG_variable
	.long	.Ldebug_loc8                    # DW_AT_location
	.long	.Linfo_string77                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	120                             # DW_AT_decl_line
	.long	2040                            # DW_AT_type
	.byte	13                              # Abbrev [13] 0x1bd:0xf DW_TAG_variable
	.long	.Ldebug_loc9                    # DW_AT_location
	.long	.Linfo_string78                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	127                             # DW_AT_decl_line
	.long	1468                            # DW_AT_type
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
	.byte	89                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0x243:0xf DW_TAG_variable
	.long	.Ldebug_loc6                    # DW_AT_location
	.long	.Linfo_string75                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	90                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	15                              # Abbrev [15] 0x252:0xb DW_TAG_variable
	.long	.Linfo_string83                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	88                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	18                              # Abbrev [18] 0x25d:0x14 DW_TAG_inlined_subroutine
	.long	258                             # DW_AT_abstract_origin
	.quad	.Ltmp25                         # DW_AT_low_pc
	.long	.Ltmp26-.Ltmp25                 # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	88                              # DW_AT_call_line
	.byte	29                              # DW_AT_call_column
	.byte	16                              # Abbrev [16] 0x271:0x48 DW_TAG_lexical_block
	.long	.Ldebug_ranges4                 # DW_AT_ranges
	.byte	15                              # Abbrev [15] 0x276:0xb DW_TAG_variable
	.long	.Linfo_string79                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	95                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	16                              # Abbrev [16] 0x281:0x37 DW_TAG_lexical_block
	.long	.Ldebug_ranges5                 # DW_AT_ranges
	.byte	15                              # Abbrev [15] 0x286:0xb DW_TAG_variable
	.long	.Linfo_string84                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	96                              # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	16                              # Abbrev [16] 0x291:0x26 DW_TAG_lexical_block
	.long	.Ldebug_ranges6                 # DW_AT_ranges
	.byte	13                              # Abbrev [13] 0x296:0xf DW_TAG_variable
	.long	.Ldebug_loc7                    # DW_AT_location
	.long	.Linfo_string76                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	99                              # DW_AT_decl_line
	.long	2040                            # DW_AT_type
	.byte	16                              # Abbrev [16] 0x2a5:0x11 DW_TAG_lexical_block
	.long	.Ldebug_ranges7                 # DW_AT_ranges
	.byte	15                              # Abbrev [15] 0x2aa:0xb DW_TAG_variable
	.long	.Linfo_string80                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	108                             # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	18                              # Abbrev [18] 0x2b9:0x14 DW_TAG_inlined_subroutine
	.long	258                             # DW_AT_abstract_origin
	.quad	.Ltmp32                         # DW_AT_low_pc
	.long	.Ltmp34-.Ltmp32                 # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	89                              # DW_AT_call_line
	.byte	29                              # DW_AT_call_column
	.byte	18                              # Abbrev [18] 0x2cd:0x14 DW_TAG_inlined_subroutine
	.long	258                             # DW_AT_abstract_origin
	.quad	.Ltmp44                         # DW_AT_low_pc
	.long	.Ltmp45-.Ltmp44                 # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	90                              # DW_AT_call_line
	.byte	29                              # DW_AT_call_column
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	14                              # Abbrev [14] 0x2e5:0x33 DW_TAG_lexical_block
	.quad	.Ltmp85                         # DW_AT_low_pc
	.long	.Ltmp100-.Ltmp85                # DW_AT_high_pc
	.byte	17                              # Abbrev [17] 0x2f2:0xc DW_TAG_variable
	.byte	0                               # DW_AT_const_value
	.long	.Linfo_string79                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	135                             # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	14                              # Abbrev [14] 0x2fe:0x19 DW_TAG_lexical_block
	.quad	.Ltmp85                         # DW_AT_low_pc
	.long	.Ltmp95-.Ltmp85                 # DW_AT_high_pc
	.byte	15                              # Abbrev [15] 0x30b:0xb DW_TAG_variable
	.long	.Linfo_string80                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	137                             # DW_AT_decl_line
	.long	293                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x318:0x1d DW_TAG_GNU_call_site
	.long	1261                            # DW_AT_abstract_origin
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
	.long	1261                            # DW_AT_abstract_origin
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
	.long	1261                            # DW_AT_abstract_origin
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
	.long	1303                            # DW_AT_abstract_origin
	.quad	.Ltmp17                         # DW_AT_low_pc
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
	.asciz	"\310"
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x389:0xd DW_TAG_GNU_call_site
	.long	1415                            # DW_AT_abstract_origin
	.quad	.Ltmp74                         # DW_AT_low_pc
	.byte	19                              # Abbrev [19] 0x396:0x1a DW_TAG_GNU_call_site
	.long	1303                            # DW_AT_abstract_origin
	.quad	.Ltmp76                         # DW_AT_low_pc
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
	.ascii	"\230\001"
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x3b0:0x1b DW_TAG_GNU_call_site
	.long	1439                            # DW_AT_abstract_origin
	.quad	.Ltmp78                         # DW_AT_low_pc
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
	.long	1958                            # DW_AT_abstract_origin
	.quad	.Ltmp79                         # DW_AT_low_pc
	.byte	19                              # Abbrev [19] 0x3d8:0x15 DW_TAG_GNU_call_site
	.long	1439                            # DW_AT_abstract_origin
	.quad	.Ltmp83                         # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x3e5:0x7 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	81
	.byte	3                               # DW_AT_GNU_call_site_value
	.byte	16
	.ascii	"\210'"
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x3ed:0x14 DW_TAG_GNU_call_site
	.long	1439                            # DW_AT_abstract_origin
	.quad	.Ltmp86                         # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x3fa:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x401:0x14 DW_TAG_GNU_call_site
	.long	1439                            # DW_AT_abstract_origin
	.quad	.Ltmp87                         # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x40e:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x415:0x14 DW_TAG_GNU_call_site
	.long	1439                            # DW_AT_abstract_origin
	.quad	.Ltmp88                         # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x422:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x429:0x14 DW_TAG_GNU_call_site
	.long	1439                            # DW_AT_abstract_origin
	.quad	.Ltmp89                         # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x436:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x43d:0x14 DW_TAG_GNU_call_site
	.long	1439                            # DW_AT_abstract_origin
	.quad	.Ltmp90                         # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x44a:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x451:0x14 DW_TAG_GNU_call_site
	.long	1439                            # DW_AT_abstract_origin
	.quad	.Ltmp91                         # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x45e:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x465:0x14 DW_TAG_GNU_call_site
	.long	1439                            # DW_AT_abstract_origin
	.quad	.Ltmp92                         # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x472:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x479:0x14 DW_TAG_GNU_call_site
	.long	1439                            # DW_AT_abstract_origin
	.quad	.Ltmp93                         # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x486:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x48d:0x14 DW_TAG_GNU_call_site
	.long	1981                            # DW_AT_abstract_origin
	.quad	.Ltmp101                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x49a:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	124
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x4a1:0x14 DW_TAG_GNU_call_site
	.long	1998                            # DW_AT_abstract_origin
	.quad	.Ltmp102                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x4ae:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	115
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x4b5:0x16 DW_TAG_GNU_call_site
	.long	1998                            # DW_AT_abstract_origin
	.quad	.Ltmp103                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x4c2:0x8 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	4                               # DW_AT_GNU_call_site_value
	.byte	145
	.byte	32
	.byte	148
	.byte	8
	.byte	0                               # End Of Children Mark
	.byte	19                              # Abbrev [19] 0x4cb:0x14 DW_TAG_GNU_call_site
	.long	1998                            # DW_AT_abstract_origin
	.quad	.Ltmp104                        # DW_AT_low_pc
	.byte	20                              # Abbrev [20] 0x4d8:0x6 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_GNU_call_site_value
	.byte	118
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	21                              # Abbrev [21] 0x4df:0xd DW_TAG_GNU_call_site
	.long	1415                            # DW_AT_abstract_origin
	.quad	.Ltmp108                        # DW_AT_low_pc
	.byte	0                               # End Of Children Mark
	.byte	22                              # Abbrev [22] 0x4ed:0x17 DW_TAG_subprogram
	.long	.Linfo_string10                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	605                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1284                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	23                              # Abbrev [23] 0x4f9:0x5 DW_TAG_formal_parameter
	.long	1285                            # DW_AT_type
	.byte	23                              # Abbrev [23] 0x4fe:0x5 DW_TAG_formal_parameter
	.long	1285                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	24                              # Abbrev [24] 0x504:0x1 DW_TAG_pointer_type
	.byte	25                              # Abbrev [25] 0x505:0xb DW_TAG_typedef
	.long	1296                            # DW_AT_type
	.long	.Linfo_string12                 # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	62                              # DW_AT_decl_line
	.byte	5                               # Abbrev [5] 0x510:0x7 DW_TAG_base_type
	.long	.Linfo_string11                 # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	22                              # Abbrev [22] 0x517:0x17 DW_TAG_subprogram
	.long	.Linfo_string13                 # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.short	288                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	293                             # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	23                              # Abbrev [23] 0x523:0x5 DW_TAG_formal_parameter
	.long	1326                            # DW_AT_type
	.byte	23                              # Abbrev [23] 0x528:0x5 DW_TAG_formal_parameter
	.long	1348                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	25                              # Abbrev [25] 0x52e:0xb DW_TAG_typedef
	.long	1337                            # DW_AT_type
	.long	.Linfo_string15                 # DW_AT_name
	.byte	6                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0x539:0xb DW_TAG_typedef
	.long	293                             # DW_AT_type
	.long	.Linfo_string14                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	169                             # DW_AT_decl_line
	.byte	26                              # Abbrev [26] 0x544:0x5 DW_TAG_pointer_type
	.long	1353                            # DW_AT_type
	.byte	27                              # Abbrev [27] 0x549:0x21 DW_TAG_structure_type
	.long	.Linfo_string21                 # DW_AT_name
	.byte	16                              # DW_AT_byte_size
	.byte	7                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.byte	28                              # Abbrev [28] 0x551:0xc DW_TAG_member
	.long	.Linfo_string16                 # DW_AT_name
	.long	1386                            # DW_AT_type
	.byte	7                               # DW_AT_decl_file
	.byte	16                              # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x55d:0xc DW_TAG_member
	.long	.Linfo_string19                 # DW_AT_name
	.long	1404                            # DW_AT_type
	.byte	7                               # DW_AT_decl_file
	.byte	21                              # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	25                              # Abbrev [25] 0x56a:0xb DW_TAG_typedef
	.long	1397                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	160                             # DW_AT_decl_line
	.byte	5                               # Abbrev [5] 0x575:0x7 DW_TAG_base_type
	.long	.Linfo_string17                 # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	25                              # Abbrev [25] 0x57c:0xb DW_TAG_typedef
	.long	1397                            # DW_AT_type
	.long	.Linfo_string20                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	197                             # DW_AT_decl_line
	.byte	29                              # Abbrev [29] 0x587:0xe DW_TAG_subprogram
	.long	.Linfo_string22                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.short	804                             # DW_AT_decl_line
                                        # DW_AT_prototyped
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	23                              # Abbrev [23] 0x58f:0x5 DW_TAG_formal_parameter
	.long	1429                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	26                              # Abbrev [26] 0x595:0x5 DW_TAG_pointer_type
	.long	1434                            # DW_AT_type
	.byte	30                              # Abbrev [30] 0x59a:0x5 DW_TAG_const_type
	.long	75                              # DW_AT_type
	.byte	22                              # Abbrev [22] 0x59f:0x18 DW_TAG_subprogram
	.long	.Linfo_string23                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.short	350                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	293                             # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	23                              # Abbrev [23] 0x5ab:0x5 DW_TAG_formal_parameter
	.long	1463                            # DW_AT_type
	.byte	23                              # Abbrev [23] 0x5b0:0x5 DW_TAG_formal_parameter
	.long	1953                            # DW_AT_type
	.byte	31                              # Abbrev [31] 0x5b5:0x1 DW_TAG_unspecified_parameters
	.byte	0                               # End Of Children Mark
	.byte	32                              # Abbrev [32] 0x5b7:0x5 DW_TAG_restrict_type
	.long	1468                            # DW_AT_type
	.byte	26                              # Abbrev [26] 0x5bc:0x5 DW_TAG_pointer_type
	.long	1473                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x5c1:0xb DW_TAG_typedef
	.long	1484                            # DW_AT_type
	.long	.Linfo_string62                 # DW_AT_name
	.byte	10                              # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	27                              # Abbrev [27] 0x5cc:0x165 DW_TAG_structure_type
	.long	.Linfo_string61                 # DW_AT_name
	.byte	216                             # DW_AT_byte_size
	.byte	9                               # DW_AT_decl_file
	.byte	49                              # DW_AT_decl_line
	.byte	28                              # Abbrev [28] 0x5d4:0xc DW_TAG_member
	.long	.Linfo_string24                 # DW_AT_name
	.long	293                             # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	51                              # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x5e0:0xc DW_TAG_member
	.long	.Linfo_string25                 # DW_AT_name
	.long	1841                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	54                              # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x5ec:0xc DW_TAG_member
	.long	.Linfo_string26                 # DW_AT_name
	.long	1841                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	55                              # DW_AT_decl_line
	.byte	16                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x5f8:0xc DW_TAG_member
	.long	.Linfo_string27                 # DW_AT_name
	.long	1841                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	56                              # DW_AT_decl_line
	.byte	24                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x604:0xc DW_TAG_member
	.long	.Linfo_string28                 # DW_AT_name
	.long	1841                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	57                              # DW_AT_decl_line
	.byte	32                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x610:0xc DW_TAG_member
	.long	.Linfo_string29                 # DW_AT_name
	.long	1841                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	58                              # DW_AT_decl_line
	.byte	40                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x61c:0xc DW_TAG_member
	.long	.Linfo_string30                 # DW_AT_name
	.long	1841                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	59                              # DW_AT_decl_line
	.byte	48                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x628:0xc DW_TAG_member
	.long	.Linfo_string31                 # DW_AT_name
	.long	1841                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	60                              # DW_AT_decl_line
	.byte	56                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x634:0xc DW_TAG_member
	.long	.Linfo_string32                 # DW_AT_name
	.long	1841                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	61                              # DW_AT_decl_line
	.byte	64                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x640:0xc DW_TAG_member
	.long	.Linfo_string33                 # DW_AT_name
	.long	1841                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	64                              # DW_AT_decl_line
	.byte	72                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x64c:0xc DW_TAG_member
	.long	.Linfo_string34                 # DW_AT_name
	.long	1841                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	65                              # DW_AT_decl_line
	.byte	80                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x658:0xc DW_TAG_member
	.long	.Linfo_string35                 # DW_AT_name
	.long	1841                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	66                              # DW_AT_decl_line
	.byte	88                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x664:0xc DW_TAG_member
	.long	.Linfo_string36                 # DW_AT_name
	.long	1846                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	68                              # DW_AT_decl_line
	.byte	96                              # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x670:0xc DW_TAG_member
	.long	.Linfo_string38                 # DW_AT_name
	.long	1856                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	70                              # DW_AT_decl_line
	.byte	104                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x67c:0xc DW_TAG_member
	.long	.Linfo_string39                 # DW_AT_name
	.long	293                             # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	72                              # DW_AT_decl_line
	.byte	112                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x688:0xc DW_TAG_member
	.long	.Linfo_string40                 # DW_AT_name
	.long	293                             # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	73                              # DW_AT_decl_line
	.byte	116                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x694:0xc DW_TAG_member
	.long	.Linfo_string41                 # DW_AT_name
	.long	1861                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	74                              # DW_AT_decl_line
	.byte	120                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6a0:0xc DW_TAG_member
	.long	.Linfo_string43                 # DW_AT_name
	.long	1872                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	77                              # DW_AT_decl_line
	.byte	128                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6ac:0xc DW_TAG_member
	.long	.Linfo_string45                 # DW_AT_name
	.long	1879                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	78                              # DW_AT_decl_line
	.byte	130                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6b8:0xc DW_TAG_member
	.long	.Linfo_string47                 # DW_AT_name
	.long	1886                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	79                              # DW_AT_decl_line
	.byte	131                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6c4:0xc DW_TAG_member
	.long	.Linfo_string48                 # DW_AT_name
	.long	1898                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	81                              # DW_AT_decl_line
	.byte	136                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6d0:0xc DW_TAG_member
	.long	.Linfo_string50                 # DW_AT_name
	.long	1910                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	89                              # DW_AT_decl_line
	.byte	144                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6dc:0xc DW_TAG_member
	.long	.Linfo_string52                 # DW_AT_name
	.long	1921                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	91                              # DW_AT_decl_line
	.byte	152                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6e8:0xc DW_TAG_member
	.long	.Linfo_string54                 # DW_AT_name
	.long	1931                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	92                              # DW_AT_decl_line
	.byte	160                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x6f4:0xc DW_TAG_member
	.long	.Linfo_string56                 # DW_AT_name
	.long	1856                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	93                              # DW_AT_decl_line
	.byte	168                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x700:0xc DW_TAG_member
	.long	.Linfo_string57                 # DW_AT_name
	.long	1284                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	94                              # DW_AT_decl_line
	.byte	176                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x70c:0xc DW_TAG_member
	.long	.Linfo_string58                 # DW_AT_name
	.long	1285                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	95                              # DW_AT_decl_line
	.byte	184                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x718:0xc DW_TAG_member
	.long	.Linfo_string59                 # DW_AT_name
	.long	293                             # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	96                              # DW_AT_decl_line
	.byte	192                             # DW_AT_data_member_location
	.byte	28                              # Abbrev [28] 0x724:0xc DW_TAG_member
	.long	.Linfo_string60                 # DW_AT_name
	.long	1941                            # DW_AT_type
	.byte	9                               # DW_AT_decl_file
	.byte	98                              # DW_AT_decl_line
	.byte	196                             # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	26                              # Abbrev [26] 0x731:0x5 DW_TAG_pointer_type
	.long	75                              # DW_AT_type
	.byte	26                              # Abbrev [26] 0x736:0x5 DW_TAG_pointer_type
	.long	1851                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x73b:0x5 DW_TAG_structure_type
	.long	.Linfo_string37                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	26                              # Abbrev [26] 0x740:0x5 DW_TAG_pointer_type
	.long	1484                            # DW_AT_type
	.byte	25                              # Abbrev [25] 0x745:0xb DW_TAG_typedef
	.long	1397                            # DW_AT_type
	.long	.Linfo_string42                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	152                             # DW_AT_decl_line
	.byte	5                               # Abbrev [5] 0x750:0x7 DW_TAG_base_type
	.long	.Linfo_string44                 # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	2                               # DW_AT_byte_size
	.byte	5                               # Abbrev [5] 0x757:0x7 DW_TAG_base_type
	.long	.Linfo_string46                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x75e:0xc DW_TAG_array_type
	.long	75                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x763:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	26                              # Abbrev [26] 0x76a:0x5 DW_TAG_pointer_type
	.long	1903                            # DW_AT_type
	.byte	34                              # Abbrev [34] 0x76f:0x7 DW_TAG_typedef
	.long	.Linfo_string49                 # DW_AT_name
	.byte	9                               # DW_AT_decl_file
	.byte	43                              # DW_AT_decl_line
	.byte	25                              # Abbrev [25] 0x776:0xb DW_TAG_typedef
	.long	1397                            # DW_AT_type
	.long	.Linfo_string51                 # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	153                             # DW_AT_decl_line
	.byte	26                              # Abbrev [26] 0x781:0x5 DW_TAG_pointer_type
	.long	1926                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x786:0x5 DW_TAG_structure_type
	.long	.Linfo_string53                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	26                              # Abbrev [26] 0x78b:0x5 DW_TAG_pointer_type
	.long	1936                            # DW_AT_type
	.byte	33                              # Abbrev [33] 0x790:0x5 DW_TAG_structure_type
	.long	.Linfo_string55                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	3                               # Abbrev [3] 0x795:0xc DW_TAG_array_type
	.long	75                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x79a:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	20                              # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	32                              # Abbrev [32] 0x7a1:0x5 DW_TAG_restrict_type
	.long	1429                            # DW_AT_type
	.byte	22                              # Abbrev [22] 0x7a6:0x17 DW_TAG_subprogram
	.long	.Linfo_string63                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.short	258                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	1468                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	23                              # Abbrev [23] 0x7b2:0x5 DW_TAG_formal_parameter
	.long	1953                            # DW_AT_type
	.byte	23                              # Abbrev [23] 0x7b7:0x5 DW_TAG_formal_parameter
	.long	1953                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	35                              # Abbrev [35] 0x7bd:0x11 DW_TAG_subprogram
	.long	.Linfo_string64                 # DW_AT_name
	.byte	8                               # DW_AT_decl_file
	.byte	178                             # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	293                             # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	23                              # Abbrev [23] 0x7c8:0x5 DW_TAG_formal_parameter
	.long	1468                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	29                              # Abbrev [29] 0x7ce:0xe DW_TAG_subprogram
	.long	.Linfo_string65                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.short	568                             # DW_AT_decl_line
                                        # DW_AT_prototyped
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	23                              # Abbrev [23] 0x7d6:0x5 DW_TAG_formal_parameter
	.long	1284                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	26                              # Abbrev [26] 0x7dc:0x5 DW_TAG_pointer_type
	.long	1841                            # DW_AT_type
	.byte	32                              # Abbrev [32] 0x7e1:0x5 DW_TAG_restrict_type
	.long	2022                            # DW_AT_type
	.byte	26                              # Abbrev [26] 0x7e6:0x5 DW_TAG_pointer_type
	.long	2027                            # DW_AT_type
	.byte	3                               # Abbrev [3] 0x7eb:0xd DW_TAG_array_type
	.long	2040                            # DW_AT_type
	.byte	36                              # Abbrev [36] 0x7f0:0x7 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.short	5000                            # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x7f8:0x7 DW_TAG_base_type
	.long	.Linfo_string71                 # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges4:
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges5:
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges6:
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges7:
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang based Intel(R) oneAPI DPC++/C++ Compiler 2023.2.0 (2023.2.0.20230721)" # string offset=0
.Linfo_string1:
	.asciz	" --intel -g -O3 -S -D ENABLE_TIMING src/matmul_seq_tile.c -o matmul_seq_tile_O3.s -fveclib=SVML -fheinous-gnu-extensions" # string offset=76
.Linfo_string2:
	.asciz	"src/matmul_seq_tile.c"         # string offset=197
.Linfo_string3:
	.asciz	"/home/Zsf/Desktop/_FINAL_PROJECT" # string offset=219
.Linfo_string4:
	.asciz	"char"                          # string offset=252
.Linfo_string5:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=257
.Linfo_string6:
	.asciz	"min"                           # string offset=277
.Linfo_string7:
	.asciz	"int"                           # string offset=281
.Linfo_string8:
	.asciz	"a"                             # string offset=285
.Linfo_string9:
	.asciz	"b"                             # string offset=287
.Linfo_string10:
	.asciz	"aligned_alloc"                 # string offset=289
.Linfo_string11:
	.asciz	"unsigned long"                 # string offset=303
.Linfo_string12:
	.asciz	"size_t"                        # string offset=317
.Linfo_string13:
	.asciz	"clock_gettime"                 # string offset=324
.Linfo_string14:
	.asciz	"__clockid_t"                   # string offset=338
.Linfo_string15:
	.asciz	"clockid_t"                     # string offset=350
.Linfo_string16:
	.asciz	"tv_sec"                        # string offset=360
.Linfo_string17:
	.asciz	"long"                          # string offset=367
.Linfo_string18:
	.asciz	"__time_t"                      # string offset=372
.Linfo_string19:
	.asciz	"tv_nsec"                       # string offset=381
.Linfo_string20:
	.asciz	"__syscall_slong_t"             # string offset=389
.Linfo_string21:
	.asciz	"timespec"                      # string offset=407
.Linfo_string22:
	.asciz	"perror"                        # string offset=416
.Linfo_string23:
	.asciz	"fprintf"                       # string offset=423
.Linfo_string24:
	.asciz	"_flags"                        # string offset=431
.Linfo_string25:
	.asciz	"_IO_read_ptr"                  # string offset=438
.Linfo_string26:
	.asciz	"_IO_read_end"                  # string offset=451
.Linfo_string27:
	.asciz	"_IO_read_base"                 # string offset=464
.Linfo_string28:
	.asciz	"_IO_write_base"                # string offset=478
.Linfo_string29:
	.asciz	"_IO_write_ptr"                 # string offset=493
.Linfo_string30:
	.asciz	"_IO_write_end"                 # string offset=507
.Linfo_string31:
	.asciz	"_IO_buf_base"                  # string offset=521
.Linfo_string32:
	.asciz	"_IO_buf_end"                   # string offset=534
.Linfo_string33:
	.asciz	"_IO_save_base"                 # string offset=546
.Linfo_string34:
	.asciz	"_IO_backup_base"               # string offset=560
.Linfo_string35:
	.asciz	"_IO_save_end"                  # string offset=576
.Linfo_string36:
	.asciz	"_markers"                      # string offset=589
.Linfo_string37:
	.asciz	"_IO_marker"                    # string offset=598
.Linfo_string38:
	.asciz	"_chain"                        # string offset=609
.Linfo_string39:
	.asciz	"_fileno"                       # string offset=616
.Linfo_string40:
	.asciz	"_flags2"                       # string offset=624
.Linfo_string41:
	.asciz	"_old_offset"                   # string offset=632
.Linfo_string42:
	.asciz	"__off_t"                       # string offset=644
.Linfo_string43:
	.asciz	"_cur_column"                   # string offset=652
.Linfo_string44:
	.asciz	"unsigned short"                # string offset=664
.Linfo_string45:
	.asciz	"_vtable_offset"                # string offset=679
.Linfo_string46:
	.asciz	"signed char"                   # string offset=694
.Linfo_string47:
	.asciz	"_shortbuf"                     # string offset=706
.Linfo_string48:
	.asciz	"_lock"                         # string offset=716
.Linfo_string49:
	.asciz	"_IO_lock_t"                    # string offset=722
.Linfo_string50:
	.asciz	"_offset"                       # string offset=733
.Linfo_string51:
	.asciz	"__off64_t"                     # string offset=741
.Linfo_string52:
	.asciz	"_codecvt"                      # string offset=751
.Linfo_string53:
	.asciz	"_IO_codecvt"                   # string offset=760
.Linfo_string54:
	.asciz	"_wide_data"                    # string offset=772
.Linfo_string55:
	.asciz	"_IO_wide_data"                 # string offset=783
.Linfo_string56:
	.asciz	"_freeres_list"                 # string offset=797
.Linfo_string57:
	.asciz	"_freeres_buf"                  # string offset=811
.Linfo_string58:
	.asciz	"__pad5"                        # string offset=824
.Linfo_string59:
	.asciz	"_mode"                         # string offset=831
.Linfo_string60:
	.asciz	"_unused2"                      # string offset=837
.Linfo_string61:
	.asciz	"_IO_FILE"                      # string offset=846
.Linfo_string62:
	.asciz	"FILE"                          # string offset=855
.Linfo_string63:
	.asciz	"fopen"                         # string offset=860
.Linfo_string64:
	.asciz	"fclose"                        # string offset=866
.Linfo_string65:
	.asciz	"free"                          # string offset=873
.Linfo_string66:
	.asciz	"main"                          # string offset=878
.Linfo_string67:
	.asciz	"start"                         # string offset=883
.Linfo_string68:
	.asciz	"end"                           # string offset=889
.Linfo_string69:
	.asciz	"argc"                          # string offset=893
.Linfo_string70:
	.asciz	"argv"                          # string offset=898
.Linfo_string71:
	.asciz	"double"                        # string offset=903
.Linfo_string72:
	.asciz	"c"                             # string offset=910
.Linfo_string73:
	.asciz	"ii"                            # string offset=912
.Linfo_string74:
	.asciz	"k_max"                         # string offset=915
.Linfo_string75:
	.asciz	"j_max"                         # string offset=921
.Linfo_string76:
	.asciz	"r"                             # string offset=927
.Linfo_string77:
	.asciz	"time_taken"                    # string offset=929
.Linfo_string78:
	.asciz	"f"                             # string offset=940
.Linfo_string79:
	.asciz	"i"                             # string offset=942
.Linfo_string80:
	.asciz	"j"                             # string offset=944
.Linfo_string81:
	.asciz	"kk"                            # string offset=946
.Linfo_string82:
	.asciz	"jj"                            # string offset=949
.Linfo_string83:
	.asciz	"i_max"                         # string offset=952
.Linfo_string84:
	.asciz	"k"                             # string offset=958
	.ident	"Intel(R) oneAPI DPC++/C++ Compiler 2023.2.0 (2023.2.0.20230721)"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
