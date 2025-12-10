
bin/seq_bench/icc/matmul_O2:     file format elf64-x86-64


Disassembly of section .init:

0000000000401000 <_init>:
  401000:	48 83 ec 08          	sub    $0x8,%rsp
  401004:	48 8b 05 b5 5f 00 00 	mov    0x5fb5(%rip),%rax        # 406fc0 <__gmon_start__@Base>
  40100b:	48 85 c0             	test   %rax,%rax
  40100e:	74 02                	je     401012 <_init+0x12>
  401010:	ff d0                	call   *%rax
  401012:	48 83 c4 08          	add    $0x8,%rsp
  401016:	c3                   	ret

Disassembly of section .plt:

0000000000401020 <getenv@plt-0x10>:
  401020:	ff 35 ca 5f 00 00    	push   0x5fca(%rip)        # 406ff0 <_GLOBAL_OFFSET_TABLE_+0x8>
  401026:	ff 25 cc 5f 00 00    	jmp    *0x5fcc(%rip)        # 406ff8 <_GLOBAL_OFFSET_TABLE_+0x10>
  40102c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401030 <getenv@plt>:
  401030:	ff 25 ca 5f 00 00    	jmp    *0x5fca(%rip)        # 407000 <getenv@GLIBC_2.2.5>
  401036:	68 00 00 00 00       	push   $0x0
  40103b:	e9 e0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401040 <free@plt>:
  401040:	ff 25 c2 5f 00 00    	jmp    *0x5fc2(%rip)        # 407008 <free@GLIBC_2.2.5>
  401046:	68 01 00 00 00       	push   $0x1
  40104b:	e9 d0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401050 <setenv@plt>:
  401050:	ff 25 ba 5f 00 00    	jmp    *0x5fba(%rip)        # 407010 <setenv@GLIBC_2.2.5>
  401056:	68 02 00 00 00       	push   $0x2
  40105b:	e9 c0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401060 <clock@plt>:
  401060:	ff 25 b2 5f 00 00    	jmp    *0x5fb2(%rip)        # 407018 <clock@GLIBC_2.2.5>
  401066:	68 03 00 00 00       	push   $0x3
  40106b:	e9 b0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401070 <fclose@plt>:
  401070:	ff 25 aa 5f 00 00    	jmp    *0x5faa(%rip)        # 407020 <fclose@GLIBC_2.2.5>
  401076:	68 04 00 00 00       	push   $0x4
  40107b:	e9 a0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401080 <strlen@plt>:
  401080:	ff 25 a2 5f 00 00    	jmp    *0x5fa2(%rip)        # 407028 <strlen@GLIBC_2.2.5>
  401086:	68 05 00 00 00       	push   $0x5
  40108b:	e9 90 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401090 <__stack_chk_fail@plt>:
  401090:	ff 25 9a 5f 00 00    	jmp    *0x5f9a(%rip)        # 407030 <__stack_chk_fail@GLIBC_2.4>
  401096:	68 06 00 00 00       	push   $0x6
  40109b:	e9 80 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010a0 <strchr@plt>:
  4010a0:	ff 25 92 5f 00 00    	jmp    *0x5f92(%rip)        # 407038 <strchr@GLIBC_2.2.5>
  4010a6:	68 07 00 00 00       	push   $0x7
  4010ab:	e9 70 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010b0 <memset@plt>:
  4010b0:	ff 25 8a 5f 00 00    	jmp    *0x5f8a(%rip)        # 407040 <memset@GLIBC_2.2.5>
  4010b6:	68 08 00 00 00       	push   $0x8
  4010bb:	e9 60 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010c0 <fputc@plt>:
  4010c0:	ff 25 82 5f 00 00    	jmp    *0x5f82(%rip)        # 407048 <fputc@GLIBC_2.2.5>
  4010c6:	68 09 00 00 00       	push   $0x9
  4010cb:	e9 50 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010d0 <fprintf@plt>:
  4010d0:	ff 25 7a 5f 00 00    	jmp    *0x5f7a(%rip)        # 407050 <fprintf@GLIBC_2.2.5>
  4010d6:	68 0a 00 00 00       	push   $0xa
  4010db:	e9 40 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010e0 <malloc@plt>:
  4010e0:	ff 25 72 5f 00 00    	jmp    *0x5f72(%rip)        # 407058 <malloc@GLIBC_2.2.5>
  4010e6:	68 0b 00 00 00       	push   $0xb
  4010eb:	e9 30 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010f0 <catopen@plt>:
  4010f0:	ff 25 6a 5f 00 00    	jmp    *0x5f6a(%rip)        # 407060 <catopen@GLIBC_2.2.5>
  4010f6:	68 0c 00 00 00       	push   $0xc
  4010fb:	e9 20 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401100 <__vsnprintf_chk@plt>:
  401100:	ff 25 62 5f 00 00    	jmp    *0x5f62(%rip)        # 407068 <__vsnprintf_chk@GLIBC_2.3.4>
  401106:	68 0d 00 00 00       	push   $0xd
  40110b:	e9 10 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401110 <__strncpy_chk@plt>:
  401110:	ff 25 5a 5f 00 00    	jmp    *0x5f5a(%rip)        # 407070 <__strncpy_chk@GLIBC_2.3.4>
  401116:	68 0e 00 00 00       	push   $0xe
  40111b:	e9 00 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401120 <__printf_chk@plt>:
  401120:	ff 25 52 5f 00 00    	jmp    *0x5f52(%rip)        # 407078 <__printf_chk@GLIBC_2.3.4>
  401126:	68 0f 00 00 00       	push   $0xf
  40112b:	e9 f0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401130 <fopen@plt>:
  401130:	ff 25 4a 5f 00 00    	jmp    *0x5f4a(%rip)        # 407080 <fopen@GLIBC_2.2.5>
  401136:	68 10 00 00 00       	push   $0x10
  40113b:	e9 e0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401140 <perror@plt>:
  401140:	ff 25 42 5f 00 00    	jmp    *0x5f42(%rip)        # 407088 <perror@GLIBC_2.2.5>
  401146:	68 11 00 00 00       	push   $0x11
  40114b:	e9 d0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401150 <catgets@plt>:
  401150:	ff 25 3a 5f 00 00    	jmp    *0x5f3a(%rip)        # 407090 <catgets@GLIBC_2.2.5>
  401156:	68 12 00 00 00       	push   $0x12
  40115b:	e9 c0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401160 <exit@plt>:
  401160:	ff 25 32 5f 00 00    	jmp    *0x5f32(%rip)        # 407098 <exit@GLIBC_2.2.5>
  401166:	68 13 00 00 00       	push   $0x13
  40116b:	e9 b0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401170 <__fprintf_chk@plt>:
  401170:	ff 25 2a 5f 00 00    	jmp    *0x5f2a(%rip)        # 4070a0 <__fprintf_chk@GLIBC_2.3.4>
  401176:	68 14 00 00 00       	push   $0x14
  40117b:	e9 a0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401180 <__strncat_chk@plt>:
  401180:	ff 25 22 5f 00 00    	jmp    *0x5f22(%rip)        # 4070a8 <__strncat_chk@GLIBC_2.3.4>
  401186:	68 15 00 00 00       	push   $0x15
  40118b:	e9 90 fe ff ff       	jmp    401020 <_init+0x20>

Disassembly of section .plt.got:

0000000000401190 <__cxa_finalize@plt>:
  401190:	ff 25 3a 5e 00 00    	jmp    *0x5e3a(%rip)        # 406fd0 <__cxa_finalize@GLIBC_2.2.5>
  401196:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004011a0 <_start>:
  4011a0:	31 ed                	xor    %ebp,%ebp
  4011a2:	49 89 d1             	mov    %rdx,%r9
  4011a5:	5e                   	pop    %rsi
  4011a6:	48 89 e2             	mov    %rsp,%rdx
  4011a9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4011ad:	50                   	push   %rax
  4011ae:	54                   	push   %rsp
  4011af:	45 31 c0             	xor    %r8d,%r8d
  4011b2:	31 c9                	xor    %ecx,%ecx
  4011b4:	48 c7 c7 90 12 40 00 	mov    $0x401290,%rdi
  4011bb:	ff 15 ef 5d 00 00    	call   *0x5def(%rip)        # 406fb0 <__libc_start_main@GLIBC_2.34>
  4011c1:	f4                   	hlt
  4011c2:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4011c9:	00 00 00 
  4011cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004011d0 <deregister_tm_clones>:
  4011d0:	48 8d 3d e9 5e 00 00 	lea    0x5ee9(%rip),%rdi        # 4070c0 <stderr@GLIBC_2.2.5>
  4011d7:	48 8d 05 e2 5e 00 00 	lea    0x5ee2(%rip),%rax        # 4070c0 <stderr@GLIBC_2.2.5>
  4011de:	48 39 f8             	cmp    %rdi,%rax
  4011e1:	74 15                	je     4011f8 <deregister_tm_clones+0x28>
  4011e3:	48 8b 05 ce 5d 00 00 	mov    0x5dce(%rip),%rax        # 406fb8 <_ITM_deregisterTMCloneTable@Base>
  4011ea:	48 85 c0             	test   %rax,%rax
  4011ed:	74 09                	je     4011f8 <deregister_tm_clones+0x28>
  4011ef:	ff e0                	jmp    *%rax
  4011f1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4011f8:	c3                   	ret
  4011f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401200 <register_tm_clones>:
  401200:	48 8d 3d b9 5e 00 00 	lea    0x5eb9(%rip),%rdi        # 4070c0 <stderr@GLIBC_2.2.5>
  401207:	48 8d 35 b2 5e 00 00 	lea    0x5eb2(%rip),%rsi        # 4070c0 <stderr@GLIBC_2.2.5>
  40120e:	48 29 fe             	sub    %rdi,%rsi
  401211:	48 89 f0             	mov    %rsi,%rax
  401214:	48 c1 ee 3f          	shr    $0x3f,%rsi
  401218:	48 c1 f8 03          	sar    $0x3,%rax
  40121c:	48 01 c6             	add    %rax,%rsi
  40121f:	48 d1 fe             	sar    %rsi
  401222:	74 14                	je     401238 <register_tm_clones+0x38>
  401224:	48 8b 05 9d 5d 00 00 	mov    0x5d9d(%rip),%rax        # 406fc8 <_ITM_registerTMCloneTable@Base>
  40122b:	48 85 c0             	test   %rax,%rax
  40122e:	74 08                	je     401238 <register_tm_clones+0x38>
  401230:	ff e0                	jmp    *%rax
  401232:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401238:	c3                   	ret
  401239:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401240 <__do_global_dtors_aux>:
  401240:	f3 0f 1e fa          	endbr64
  401244:	80 3d 7d 5e 00 00 00 	cmpb   $0x0,0x5e7d(%rip)        # 4070c8 <completed.0>
  40124b:	75 2b                	jne    401278 <__do_global_dtors_aux+0x38>
  40124d:	55                   	push   %rbp
  40124e:	48 83 3d 7a 5d 00 00 	cmpq   $0x0,0x5d7a(%rip)        # 406fd0 <__cxa_finalize@GLIBC_2.2.5>
  401255:	00 
  401256:	48 89 e5             	mov    %rsp,%rbp
  401259:	74 0c                	je     401267 <__do_global_dtors_aux+0x27>
  40125b:	48 8b 3d 56 5e 00 00 	mov    0x5e56(%rip),%rdi        # 4070b8 <__dso_handle>
  401262:	e8 29 ff ff ff       	call   401190 <__cxa_finalize@plt>
  401267:	e8 64 ff ff ff       	call   4011d0 <deregister_tm_clones>
  40126c:	c6 05 55 5e 00 00 01 	movb   $0x1,0x5e55(%rip)        # 4070c8 <completed.0>
  401273:	5d                   	pop    %rbp
  401274:	c3                   	ret
  401275:	0f 1f 00             	nopl   (%rax)
  401278:	c3                   	ret
  401279:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401280 <frame_dummy>:
  401280:	f3 0f 1e fa          	endbr64
  401284:	e9 77 ff ff ff       	jmp    401200 <register_tm_clones>
  401289:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401290 <main>:
  401290:	55                   	push   %rbp
  401291:	48 89 e5             	mov    %rsp,%rbp
  401294:	48 83 e4 80          	and    $0xffffffffffffff80,%rsp
  401298:	41 54                	push   %r12
  40129a:	41 55                	push   %r13
  40129c:	41 56                	push   %r14
  40129e:	41 57                	push   %r15
  4012a0:	53                   	push   %rbx
  4012a1:	48 83 ec 58          	sub    $0x58,%rsp
  4012a5:	bf 03 00 00 00       	mov    $0x3,%edi
  4012aa:	33 f6                	xor    %esi,%esi
  4012ac:	e8 bf 07 00 00       	call   401a70 <__intel_new_feature_proc_init>
  4012b1:	0f ae 1c 24          	stmxcsr (%rsp)
  4012b5:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4012ba:	81 0c 24 40 80 00 00 	orl    $0x8040,(%rsp)
  4012c1:	0f ae 14 24          	ldmxcsr (%rsp)
  4012c5:	e8 16 fe ff ff       	call   4010e0 <malloc@plt>
  4012ca:	49 89 c6             	mov    %rax,%r14
  4012cd:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4012d2:	e8 09 fe ff ff       	call   4010e0 <malloc@plt>
  4012d7:	48 89 c3             	mov    %rax,%rbx
  4012da:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4012df:	e8 fc fd ff ff       	call   4010e0 <malloc@plt>
  4012e4:	49 89 c7             	mov    %rax,%r15
  4012e7:	4c 89 fa             	mov    %r15,%rdx
  4012ea:	48 83 e2 0f          	and    $0xf,%rdx
  4012ee:	84 d2                	test   %dl,%dl
  4012f0:	74 2c                	je     40131e <main+0x8e>
  4012f2:	f6 c2 07             	test   $0x7,%dl
  4012f5:	0f 85 f7 03 00 00    	jne    4016f2 <main+0x462>
  4012fb:	48 b8 00 00 00 00 00 	movabs $0x4000000000000000,%rax
  401302:	00 00 40 
  401305:	48 b9 00 00 00 00 00 	movabs $0x4008000000000000,%rcx
  40130c:	00 08 40 
  40130f:	49 89 06             	mov    %rax,(%r14)
  401312:	b2 01                	mov    $0x1,%dl
  401314:	48 89 0b             	mov    %rcx,(%rbx)
  401317:	49 c7 07 00 00 00 00 	movq   $0x0,(%r15)
  40131e:	0f b6 c2             	movzbl %dl,%eax
  401321:	f6 da                	neg    %dl
  401323:	80 e2 01             	and    $0x1,%dl
  401326:	0f b6 d2             	movzbl %dl,%edx
  401329:	66 0f ef c0          	pxor   %xmm0,%xmm0
  40132d:	48 f7 da             	neg    %rdx
  401330:	48 81 c2 40 78 7d 01 	add    $0x17d7840,%rdx
  401337:	48 8d 0c c3          	lea    (%rbx,%rax,8),%rcx
  40133b:	0f 10 15 ce 2c 00 00 	movups 0x2cce(%rip),%xmm2        # 404010 <_IO_stdin_used+0x10>
  401342:	0f 10 0d d7 2c 00 00 	movups 0x2cd7(%rip),%xmm1        # 404020 <_IO_stdin_used+0x20>
  401349:	48 f7 c1 0f 00 00 00 	test   $0xf,%rcx
  401350:	74 1a                	je     40136c <main+0xdc>
  401352:	41 0f 11 14 c6       	movups %xmm2,(%r14,%rax,8)
  401357:	0f 11 0c c3          	movups %xmm1,(%rbx,%rax,8)
  40135b:	66 41 0f 2b 04 c7    	movntpd %xmm0,(%r15,%rax,8)
  401361:	48 83 c0 02          	add    $0x2,%rax
  401365:	48 3b c2             	cmp    %rdx,%rax
  401368:	72 e8                	jb     401352 <main+0xc2>
  40136a:	eb 19                	jmp    401385 <main+0xf5>
  40136c:	41 0f 11 14 c6       	movups %xmm2,(%r14,%rax,8)
  401371:	66 0f 2b 0c c3       	movntpd %xmm1,(%rbx,%rax,8)
  401376:	66 41 0f 2b 04 c7    	movntpd %xmm0,(%r15,%rax,8)
  40137c:	48 83 c0 02          	add    $0x2,%rax
  401380:	48 3b c2             	cmp    %rdx,%rax
  401383:	72 e7                	jb     40136c <main+0xdc>
  401385:	0f ae f0             	mfence
  401388:	48 81 fa 40 78 7d 01 	cmp    $0x17d7840,%rdx
  40138f:	73 2e                	jae    4013bf <main+0x12f>
  401391:	48 b9 00 00 00 00 00 	movabs $0x4000000000000000,%rcx
  401398:	00 00 40 
  40139b:	48 b8 00 00 00 00 00 	movabs $0x4008000000000000,%rax
  4013a2:	00 08 40 
  4013a5:	33 f6                	xor    %esi,%esi
  4013a7:	49 89 0c d6          	mov    %rcx,(%r14,%rdx,8)
  4013ab:	48 89 04 d3          	mov    %rax,(%rbx,%rdx,8)
  4013af:	49 89 34 d7          	mov    %rsi,(%r15,%rdx,8)
  4013b3:	48 ff c2             	inc    %rdx
  4013b6:	48 81 fa 40 78 7d 01 	cmp    $0x17d7840,%rdx
  4013bd:	72 e6                	jb     4013a5 <main+0x115>
  4013bf:	e8 9c fc ff ff       	call   401060 <clock@plt>
  4013c4:	49 89 c5             	mov    %rax,%r13
  4013c7:	33 c0                	xor    %eax,%eax
  4013c9:	33 d2                	xor    %edx,%edx
  4013cb:	4c 89 2c 24          	mov    %r13,(%rsp)
  4013cf:	45 33 c0             	xor    %r8d,%r8d
  4013d2:	45 33 d2             	xor    %r10d,%r10d
  4013d5:	4d 8d 24 17          	lea    (%r15,%rdx,1),%r12
  4013d9:	4d 89 e1             	mov    %r12,%r9
  4013dc:	4d 8d 1c 16          	lea    (%r14,%rdx,1),%r11
  4013e0:	49 83 e1 0f          	and    $0xf,%r9
  4013e4:	45 33 ed             	xor    %r13d,%r13d
  4013e7:	44 89 ce             	mov    %r9d,%esi
  4013ea:	48 89 d9             	mov    %rbx,%rcx
  4013ed:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
  4013f2:	83 e6 07             	and    $0x7,%esi
  4013f5:	89 44 24 18          	mov    %eax,0x18(%rsp)
  4013f9:	4c 89 7c 24 10       	mov    %r15,0x10(%rsp)
  4013fe:	4c 89 74 24 08       	mov    %r14,0x8(%rsp)
  401403:	f2 43 0f 10 0c d3    	movsd  (%r11,%r10,8),%xmm1
  401409:	45 89 cf             	mov    %r9d,%r15d
  40140c:	45 85 c9             	test   %r9d,%r9d
  40140f:	74 25                	je     401436 <main+0x1a6>
  401411:	85 f6                	test   %esi,%esi
  401413:	0f 85 d1 02 00 00    	jne    4016ea <main+0x45a>
  401419:	f2 41 0f 10 44 1d 00 	movsd  0x0(%r13,%rbx,1),%xmm0
  401420:	41 bf 01 00 00 00    	mov    $0x1,%r15d
  401426:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  40142a:	f2 41 0f 58 04 24    	addsd  (%r12),%xmm0
  401430:	f2 41 0f 11 04 24    	movsd  %xmm0,(%r12)
  401436:	44 89 fa             	mov    %r15d,%edx
  401439:	4a 8d 04 2b          	lea    (%rbx,%r13,1),%rax
  40143d:	f7 da                	neg    %edx
  40143f:	83 e2 07             	and    $0x7,%edx
  401442:	f7 da                	neg    %edx
  401444:	81 c2 88 13 00 00    	add    $0x1388,%edx
  40144a:	41 89 d6             	mov    %edx,%r14d
  40144d:	4a 8d 3c f8          	lea    (%rax,%r15,8),%rdi
  401451:	48 f7 c7 0f 00 00 00 	test   $0xf,%rdi
  401458:	74 7a                	je     4014d4 <main+0x244>
  40145a:	0f 28 c1             	movaps %xmm1,%xmm0
  40145d:	66 0f 14 c0          	unpcklpd %xmm0,%xmm0
  401461:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401468:	00 
  401469:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  401470:	42 0f 10 14 f8       	movups (%rax,%r15,8),%xmm2
  401475:	42 0f 10 5c f8 10    	movups 0x10(%rax,%r15,8),%xmm3
  40147b:	42 0f 10 64 f8 20    	movups 0x20(%rax,%r15,8),%xmm4
  401481:	42 0f 10 6c f8 30    	movups 0x30(%rax,%r15,8),%xmm5
  401487:	66 0f 59 d0          	mulpd  %xmm0,%xmm2
  40148b:	66 0f 59 d8          	mulpd  %xmm0,%xmm3
  40148f:	66 0f 59 e0          	mulpd  %xmm0,%xmm4
  401493:	66 0f 59 e8          	mulpd  %xmm0,%xmm5
  401497:	66 43 0f 58 14 fc    	addpd  (%r12,%r15,8),%xmm2
  40149d:	66 43 0f 58 5c fc 10 	addpd  0x10(%r12,%r15,8),%xmm3
  4014a4:	66 43 0f 58 64 fc 20 	addpd  0x20(%r12,%r15,8),%xmm4
  4014ab:	66 43 0f 58 6c fc 30 	addpd  0x30(%r12,%r15,8),%xmm5
  4014b2:	43 0f 11 14 fc       	movups %xmm2,(%r12,%r15,8)
  4014b7:	43 0f 11 5c fc 10    	movups %xmm3,0x10(%r12,%r15,8)
  4014bd:	43 0f 11 64 fc 20    	movups %xmm4,0x20(%r12,%r15,8)
  4014c3:	43 0f 11 6c fc 30    	movups %xmm5,0x30(%r12,%r15,8)
  4014c9:	49 83 c7 08          	add    $0x8,%r15
  4014cd:	4d 3b fe             	cmp    %r14,%r15
  4014d0:	72 9e                	jb     401470 <main+0x1e0>
  4014d2:	eb 6e                	jmp    401542 <main+0x2b2>
  4014d4:	0f 28 c1             	movaps %xmm1,%xmm0
  4014d7:	66 0f 14 c0          	unpcklpd %xmm0,%xmm0
  4014db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4014e0:	42 0f 10 14 f8       	movups (%rax,%r15,8),%xmm2
  4014e5:	42 0f 10 5c f8 10    	movups 0x10(%rax,%r15,8),%xmm3
  4014eb:	42 0f 10 64 f8 20    	movups 0x20(%rax,%r15,8),%xmm4
  4014f1:	42 0f 10 6c f8 30    	movups 0x30(%rax,%r15,8),%xmm5
  4014f7:	66 0f 59 d0          	mulpd  %xmm0,%xmm2
  4014fb:	66 0f 59 d8          	mulpd  %xmm0,%xmm3
  4014ff:	66 0f 59 e0          	mulpd  %xmm0,%xmm4
  401503:	66 0f 59 e8          	mulpd  %xmm0,%xmm5
  401507:	66 43 0f 58 14 fc    	addpd  (%r12,%r15,8),%xmm2
  40150d:	66 43 0f 58 5c fc 10 	addpd  0x10(%r12,%r15,8),%xmm3
  401514:	66 43 0f 58 64 fc 20 	addpd  0x20(%r12,%r15,8),%xmm4
  40151b:	66 43 0f 58 6c fc 30 	addpd  0x30(%r12,%r15,8),%xmm5
  401522:	43 0f 11 14 fc       	movups %xmm2,(%r12,%r15,8)
  401527:	43 0f 11 5c fc 10    	movups %xmm3,0x10(%r12,%r15,8)
  40152d:	43 0f 11 64 fc 20    	movups %xmm4,0x20(%r12,%r15,8)
  401533:	43 0f 11 6c fc 30    	movups %xmm5,0x30(%r12,%r15,8)
  401539:	49 83 c7 08          	add    $0x8,%r15
  40153d:	4d 3b fe             	cmp    %r14,%r15
  401540:	72 9e                	jb     4014e0 <main+0x250>
  401542:	44 89 c0             	mov    %r8d,%eax
  401545:	44 8d 72 01          	lea    0x1(%rdx),%r14d
  401549:	41 81 fe 88 13 00 00 	cmp    $0x1388,%r14d
  401550:	77 31                	ja     401583 <main+0x2f3>
  401552:	41 89 d6             	mov    %edx,%r14d
  401555:	41 f7 de             	neg    %r14d
  401558:	41 81 c6 88 13 00 00 	add    $0x1388,%r14d
  40155f:	44 8d 3c 02          	lea    (%rdx,%rax,1),%r15d
  401563:	ff c0                	inc    %eax
  401565:	4d 63 ff             	movslq %r15d,%r15
  401568:	f2 42 0f 10 04 f9    	movsd  (%rcx,%r15,8),%xmm0
  40156e:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  401572:	f2 43 0f 58 04 fc    	addsd  (%r12,%r15,8),%xmm0
  401578:	f2 43 0f 11 04 fc    	movsd  %xmm0,(%r12,%r15,8)
  40157e:	41 3b c6             	cmp    %r14d,%eax
  401581:	72 dc                	jb     40155f <main+0x2cf>
  401583:	49 ff c2             	inc    %r10
  401586:	48 81 c1 40 9c 00 00 	add    $0x9c40,%rcx
  40158d:	49 81 c5 40 9c 00 00 	add    $0x9c40,%r13
  401594:	49 81 fa 88 13 00 00 	cmp    $0x1388,%r10
  40159b:	0f 82 62 fe ff ff    	jb     401403 <main+0x173>
  4015a1:	8b 44 24 18          	mov    0x18(%rsp),%eax
  4015a5:	ff c0                	inc    %eax
  4015a7:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
  4015ac:	48 81 c2 40 9c 00 00 	add    $0x9c40,%rdx
  4015b3:	4c 8b 7c 24 10       	mov    0x10(%rsp),%r15
  4015b8:	4c 8b 74 24 08       	mov    0x8(%rsp),%r14
  4015bd:	3d 88 13 00 00       	cmp    $0x1388,%eax
  4015c2:	0f 82 0a fe ff ff    	jb     4013d2 <main+0x142>
  4015c8:	4c 8b 2c 24          	mov    (%rsp),%r13
  4015cc:	e8 8f fa ff ff       	call   401060 <clock@plt>
  4015d1:	49 2b c5             	sub    %r13,%rax
  4015d4:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4015d8:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
  4015dd:	f2 0f 5e 05 5b 2a 00 	divsd  0x2a5b(%rip),%xmm0        # 404040 <_IO_stdin_used+0x40>
  4015e4:	00 
  4015e5:	be 50 40 40 00       	mov    $0x404050,%esi
  4015ea:	ba 88 13 00 00       	mov    $0x1388,%edx
  4015ef:	b8 01 00 00 00       	mov    $0x1,%eax
  4015f4:	48 8b 3d c5 5a 00 00 	mov    0x5ac5(%rip),%rdi        # 4070c0 <stderr@GLIBC_2.2.5>
  4015fb:	e8 d0 fa ff ff       	call   4010d0 <fprintf@plt>
  401600:	bf 6c 40 40 00       	mov    $0x40406c,%edi
  401605:	be 78 40 40 00       	mov    $0x404078,%esi
  40160a:	e8 21 fb ff ff       	call   401130 <fopen@plt>
  40160f:	49 89 c5             	mov    %rax,%r13
  401612:	4d 85 ed             	test   %r13,%r13
  401615:	0f 84 ae 00 00 00    	je     4016c9 <main+0x439>
  40161b:	4c 89 ef             	mov    %r13,%rdi
  40161e:	be 7c 40 40 00       	mov    $0x40407c,%esi
  401623:	ba 88 13 00 00       	mov    $0x1388,%edx
  401628:	33 c0                	xor    %eax,%eax
  40162a:	e8 a1 fa ff ff       	call   4010d0 <fprintf@plt>
  40162f:	33 d2                	xor    %edx,%edx
  401631:	4c 89 f8             	mov    %r15,%rax
  401634:	4c 89 7c 24 10       	mov    %r15,0x10(%rsp)
  401639:	49 89 c4             	mov    %rax,%r12
  40163c:	4c 89 74 24 08       	mov    %r14,0x8(%rsp)
  401641:	41 89 d6             	mov    %edx,%r14d
  401644:	45 33 ff             	xor    %r15d,%r15d
  401647:	f2 43 0f 10 04 fc    	movsd  (%r12,%r15,8),%xmm0
  40164d:	4c 89 ef             	mov    %r13,%rdi
  401650:	be 84 40 40 00       	mov    $0x404084,%esi
  401655:	b8 01 00 00 00       	mov    $0x1,%eax
  40165a:	e8 71 fa ff ff       	call   4010d0 <fprintf@plt>
  40165f:	49 ff c7             	inc    %r15
  401662:	49 81 ff e8 03 00 00 	cmp    $0x3e8,%r15
  401669:	7c dc                	jl     401647 <main+0x3b7>
  40166b:	bf 0a 00 00 00       	mov    $0xa,%edi
  401670:	4c 89 ee             	mov    %r13,%rsi
  401673:	e8 48 fa ff ff       	call   4010c0 <fputc@plt>
  401678:	41 ff c6             	inc    %r14d
  40167b:	49 81 c4 40 9c 00 00 	add    $0x9c40,%r12
  401682:	41 81 fe e8 03 00 00 	cmp    $0x3e8,%r14d
  401689:	7c b9                	jl     401644 <main+0x3b4>
  40168b:	4c 89 ef             	mov    %r13,%rdi
  40168e:	4c 8b 7c 24 10       	mov    0x10(%rsp),%r15
  401693:	4c 8b 74 24 08       	mov    0x8(%rsp),%r14
  401698:	e8 d3 f9 ff ff       	call   401070 <fclose@plt>
  40169d:	4c 89 f7             	mov    %r14,%rdi
  4016a0:	e8 9b f9 ff ff       	call   401040 <free@plt>
  4016a5:	48 89 df             	mov    %rbx,%rdi
  4016a8:	e8 93 f9 ff ff       	call   401040 <free@plt>
  4016ad:	4c 89 ff             	mov    %r15,%rdi
  4016b0:	e8 8b f9 ff ff       	call   401040 <free@plt>
  4016b5:	33 c0                	xor    %eax,%eax
  4016b7:	48 83 c4 58          	add    $0x58,%rsp
  4016bb:	5b                   	pop    %rbx
  4016bc:	41 5f                	pop    %r15
  4016be:	41 5e                	pop    %r14
  4016c0:	41 5d                	pop    %r13
  4016c2:	41 5c                	pop    %r12
  4016c4:	48 89 ec             	mov    %rbp,%rsp
  4016c7:	5d                   	pop    %rbp
  4016c8:	c3                   	ret
  4016c9:	bf 8c 40 40 00       	mov    $0x40408c,%edi
  4016ce:	e8 6d fa ff ff       	call   401140 <perror@plt>
  4016d3:	b8 01 00 00 00       	mov    $0x1,%eax
  4016d8:	48 83 c4 58          	add    $0x58,%rsp
  4016dc:	5b                   	pop    %rbx
  4016dd:	41 5f                	pop    %r15
  4016df:	41 5e                	pop    %r14
  4016e1:	41 5d                	pop    %r13
  4016e3:	41 5c                	pop    %r12
  4016e5:	48 89 ec             	mov    %rbp,%rsp
  4016e8:	5d                   	pop    %rbp
  4016e9:	c3                   	ret
  4016ea:	44 89 c2             	mov    %r8d,%edx
  4016ed:	e9 50 fe ff ff       	jmp    401542 <main+0x2b2>
  4016f2:	33 d2                	xor    %edx,%edx
  4016f4:	e9 8f fc ff ff       	jmp    401388 <main+0xf8>
  4016f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401700 <__intel_new_feature_proc_init_n>:
  401700:	f3 0f 1e fa          	endbr64
  401704:	55                   	push   %rbp
  401705:	41 57                	push   %r15
  401707:	41 56                	push   %r14
  401709:	41 55                	push   %r13
  40170b:	41 54                	push   %r12
  40170d:	53                   	push   %rbx
  40170e:	48 81 ec 38 04 00 00 	sub    $0x438,%rsp
  401715:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40171c:	00 00 
  40171e:	48 89 84 24 30 04 00 	mov    %rax,0x430(%rsp)
  401725:	00 
  401726:	0f 57 c0             	xorps  %xmm0,%xmm0
  401729:	0f 29 44 24 10       	movaps %xmm0,0x10(%rsp)
  40172e:	48 c7 c1 d0 70 40 00 	mov    $0x4070d0,%rcx
  401735:	48 83 39 00          	cmpq   $0x0,(%rcx)
  401739:	75 17                	jne    401752 <__intel_new_feature_proc_init_n+0x52>
  40173b:	e8 80 04 00 00       	call   401bc0 <__intel_cpu_features_init>
  401740:	85 c0                	test   %eax,%eax
  401742:	0f 85 0b 02 00 00    	jne    401953 <__intel_new_feature_proc_init_n+0x253>
  401748:	48 83 39 00          	cmpq   $0x0,(%rcx)
  40174c:	0f 84 01 02 00 00    	je     401953 <__intel_new_feature_proc_init_n+0x253>
  401752:	83 ff 02             	cmp    $0x2,%edi
  401755:	7d 38                	jge    40178f <__intel_new_feature_proc_init_n+0x8f>
  401757:	48 63 c7             	movslq %edi,%rax
  40175a:	48 8b 0c c1          	mov    (%rcx,%rax,8),%rcx
  40175e:	48 f7 d1             	not    %rcx
  401761:	48 85 ce             	test   %rcx,%rsi
  401764:	75 48                	jne    4017ae <__intel_new_feature_proc_init_n+0xae>
  401766:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40176d:	00 00 
  40176f:	48 3b 84 24 30 04 00 	cmp    0x430(%rsp),%rax
  401776:	00 
  401777:	0f 85 e8 02 00 00    	jne    401a65 <__intel_new_feature_proc_init_n+0x365>
  40177d:	48 81 c4 38 04 00 00 	add    $0x438,%rsp
  401784:	5b                   	pop    %rbx
  401785:	41 5c                	pop    %r12
  401787:	41 5d                	pop    %r13
  401789:	41 5e                	pop    %r14
  40178b:	41 5f                	pop    %r15
  40178d:	5d                   	pop    %rbp
  40178e:	c3                   	ret
  40178f:	bf 01 00 00 00       	mov    $0x1,%edi
  401794:	31 f6                	xor    %esi,%esi
  401796:	31 d2                	xor    %edx,%edx
  401798:	31 c0                	xor    %eax,%eax
  40179a:	e8 b1 1e 00 00       	call   403650 <__libirc_print>
  40179f:	bf 01 00 00 00       	mov    $0x1,%edi
  4017a4:	be 3a 00 00 00       	mov    $0x3a,%esi
  4017a9:	e9 bf 01 00 00       	jmp    40196d <__intel_new_feature_proc_init_n+0x26d>
  4017ae:	48 21 f1             	and    %rsi,%rcx
  4017b1:	48 89 4c c4 10       	mov    %rcx,0x10(%rsp,%rax,8)
  4017b6:	45 31 ff             	xor    %r15d,%r15d
  4017b9:	bf 39 00 00 00       	mov    $0x39,%edi
  4017be:	31 f6                	xor    %esi,%esi
  4017c0:	31 c0                	xor    %eax,%eax
  4017c2:	e8 49 1c 00 00       	call   403410 <__libirc_get_msg>
  4017c7:	48 89 04 24          	mov    %rax,(%rsp)
  4017cb:	c6 44 24 30 00       	movb   $0x0,0x30(%rsp)
  4017d0:	bd 01 00 00 00       	mov    $0x1,%ebp
  4017d5:	4c 8d 64 24 10       	lea    0x10(%rsp),%r12
  4017da:	4c 8d 6c 24 30       	lea    0x30(%rsp),%r13
  4017df:	31 db                	xor    %ebx,%ebx
  4017e1:	eb 31                	jmp    401814 <__intel_new_feature_proc_init_n+0x114>
  4017e3:	b8 ff 03 00 00       	mov    $0x3ff,%eax
  4017e8:	44 29 f8             	sub    %r15d,%eax
  4017eb:	48 63 d0             	movslq %eax,%rdx
  4017ee:	b9 00 04 00 00       	mov    $0x400,%ecx
  4017f3:	4c 89 ef             	mov    %r13,%rdi
  4017f6:	4c 89 f6             	mov    %r14,%rsi
  4017f9:	e8 82 f9 ff ff       	call   401180 <__strncat_chk@plt>
  4017fe:	4c 89 ef             	mov    %r13,%rdi
  401801:	e8 7a f8 ff ff       	call   401080 <strlen@plt>
  401806:	49 89 c7             	mov    %rax,%r15
  401809:	ff c5                	inc    %ebp
  40180b:	83 fd 47             	cmp    $0x47,%ebp
  40180e:	0f 84 26 01 00 00    	je     40193a <__intel_new_feature_proc_init_n+0x23a>
  401814:	89 e8                	mov    %ebp,%eax
  401816:	e8 95 19 00 00       	call   4031b0 <__libirc_get_feature_bitpos>
  40181b:	85 c0                	test   %eax,%eax
  40181d:	78 ea                	js     401809 <__intel_new_feature_proc_init_n+0x109>
  40181f:	4c 89 e7             	mov    %r12,%rdi
  401822:	89 ee                	mov    %ebp,%esi
  401824:	e8 c7 19 00 00       	call   4031f0 <__libirc_get_cpu_feature>
  401829:	85 c0                	test   %eax,%eax
  40182b:	74 dc                	je     401809 <__intel_new_feature_proc_init_n+0x109>
  40182d:	4c 89 e7             	mov    %r12,%rdi
  401830:	89 ee                	mov    %ebp,%esi
  401832:	e8 b9 19 00 00       	call   4031f0 <__libirc_get_cpu_feature>
  401837:	85 c0                	test   %eax,%eax
  401839:	0f 88 e6 00 00 00    	js     401925 <__intel_new_feature_proc_init_n+0x225>
  40183f:	89 ef                	mov    %ebp,%edi
  401841:	e8 7a 0e 00 00       	call   4026c0 <__libirc_get_feature_name>
  401846:	48 85 c0             	test   %rax,%rax
  401849:	0f 84 d6 00 00 00    	je     401925 <__intel_new_feature_proc_init_n+0x225>
  40184f:	49 89 c6             	mov    %rax,%r14
  401852:	80 38 00             	cmpb   $0x0,(%rax)
  401855:	0f 84 ca 00 00 00    	je     401925 <__intel_new_feature_proc_init_n+0x225>
  40185b:	80 7c 24 30 00       	cmpb   $0x0,0x30(%rsp)
  401860:	74 81                	je     4017e3 <__intel_new_feature_proc_init_n+0xe3>
  401862:	48 85 db             	test   %rbx,%rbx
  401865:	0f 84 b2 00 00 00    	je     40191d <__intel_new_feature_proc_init_n+0x21d>
  40186b:	4d 89 ec             	mov    %r13,%r12
  40186e:	48 89 df             	mov    %rbx,%rdi
  401871:	e8 0a f8 ff ff       	call   401080 <strlen@plt>
  401876:	49 89 c5             	mov    %rax,%r13
  401879:	48 8d 3d 14 28 00 00 	lea    0x2814(%rip),%rdi        # 404094 <_IO_stdin_used+0x94>
  401880:	e8 fb f7 ff ff       	call   401080 <strlen@plt>
  401885:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  40188a:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  40188f:	49 63 df             	movslq %r15d,%rbx
  401892:	48 8b 3c 24          	mov    (%rsp),%rdi
  401896:	e8 e5 f7 ff ff       	call   401080 <strlen@plt>
  40189b:	49 89 c7             	mov    %rax,%r15
  40189e:	4c 89 f7             	mov    %r14,%rdi
  4018a1:	e8 da f7 ff ff       	call   401080 <strlen@plt>
  4018a6:	49 01 dd             	add    %rbx,%r13
  4018a9:	4c 03 6c 24 28       	add    0x28(%rsp),%r13
  4018ae:	4c 01 f8             	add    %r15,%rax
  4018b1:	4c 01 e8             	add    %r13,%rax
  4018b4:	b9 ff 03 00 00       	mov    $0x3ff,%ecx
  4018b9:	29 d9                	sub    %ebx,%ecx
  4018bb:	48 63 d1             	movslq %ecx,%rdx
  4018be:	48 3d ff 03 00 00    	cmp    $0x3ff,%rax
  4018c4:	0f 87 dd 00 00 00    	ja     4019a7 <__intel_new_feature_proc_init_n+0x2a7>
  4018ca:	b9 00 04 00 00       	mov    $0x400,%ecx
  4018cf:	4d 89 e5             	mov    %r12,%r13
  4018d2:	4c 89 e7             	mov    %r12,%rdi
  4018d5:	48 8d 35 b8 27 00 00 	lea    0x27b8(%rip),%rsi        # 404094 <_IO_stdin_used+0x94>
  4018dc:	e8 9f f8 ff ff       	call   401180 <__strncat_chk@plt>
  4018e1:	4c 89 e7             	mov    %r12,%rdi
  4018e4:	e8 97 f7 ff ff       	call   401080 <strlen@plt>
  4018e9:	48 c1 e0 20          	shl    $0x20,%rax
  4018ed:	48 ba 00 00 00 00 ff 	movabs $0x3ff00000000,%rdx
  4018f4:	03 00 00 
  4018f7:	48 29 c2             	sub    %rax,%rdx
  4018fa:	48 c1 fa 20          	sar    $0x20,%rdx
  4018fe:	b9 00 04 00 00       	mov    $0x400,%ecx
  401903:	4c 89 e7             	mov    %r12,%rdi
  401906:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  40190b:	e8 70 f8 ff ff       	call   401180 <__strncat_chk@plt>
  401910:	4c 89 f3             	mov    %r14,%rbx
  401913:	4c 8d 64 24 10       	lea    0x10(%rsp),%r12
  401918:	e9 e1 fe ff ff       	jmp    4017fe <__intel_new_feature_proc_init_n+0xfe>
  40191d:	4c 89 f3             	mov    %r14,%rbx
  401920:	e9 e4 fe ff ff       	jmp    401809 <__intel_new_feature_proc_init_n+0x109>
  401925:	bf 01 00 00 00       	mov    $0x1,%edi
  40192a:	31 f6                	xor    %esi,%esi
  40192c:	31 d2                	xor    %edx,%edx
  40192e:	31 c0                	xor    %eax,%eax
  401930:	e8 1b 1d 00 00       	call   403650 <__libirc_print>
  401935:	e9 ce 00 00 00       	jmp    401a08 <__intel_new_feature_proc_init_n+0x308>
  40193a:	48 85 db             	test   %rbx,%rbx
  40193d:	0f 84 ac 00 00 00    	je     4019ef <__intel_new_feature_proc_init_n+0x2ef>
  401943:	49 89 dc             	mov    %rbx,%r12
  401946:	b8 ff 03 00 00       	mov    $0x3ff,%eax
  40194b:	44 29 f8             	sub    %r15d,%eax
  40194e:	48 63 d0             	movslq %eax,%rdx
  401951:	eb 59                	jmp    4019ac <__intel_new_feature_proc_init_n+0x2ac>
  401953:	bf 01 00 00 00       	mov    $0x1,%edi
  401958:	31 f6                	xor    %esi,%esi
  40195a:	31 d2                	xor    %edx,%edx
  40195c:	31 c0                	xor    %eax,%eax
  40195e:	e8 ed 1c 00 00       	call   403650 <__libirc_print>
  401963:	bf 01 00 00 00       	mov    $0x1,%edi
  401968:	be 3b 00 00 00       	mov    $0x3b,%esi
  40196d:	31 d2                	xor    %edx,%edx
  40196f:	31 c0                	xor    %eax,%eax
  401971:	e8 da 1c 00 00       	call   403650 <__libirc_print>
  401976:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40197d:	00 00 
  40197f:	48 3b 84 24 30 04 00 	cmp    0x430(%rsp),%rax
  401986:	00 
  401987:	0f 85 d8 00 00 00    	jne    401a65 <__intel_new_feature_proc_init_n+0x365>
  40198d:	bf 01 00 00 00       	mov    $0x1,%edi
  401992:	31 f6                	xor    %esi,%esi
  401994:	31 d2                	xor    %edx,%edx
  401996:	31 c0                	xor    %eax,%eax
  401998:	e8 b3 1c 00 00       	call   403650 <__libirc_print>
  40199d:	bf 01 00 00 00       	mov    $0x1,%edi
  4019a2:	e8 b9 f7 ff ff       	call   401160 <exit@plt>
  4019a7:	4c 8b 64 24 08       	mov    0x8(%rsp),%r12
  4019ac:	4c 8d 74 24 30       	lea    0x30(%rsp),%r14
  4019b1:	b9 00 04 00 00       	mov    $0x400,%ecx
  4019b6:	4c 89 f7             	mov    %r14,%rdi
  4019b9:	48 8b 34 24          	mov    (%rsp),%rsi
  4019bd:	e8 be f7 ff ff       	call   401180 <__strncat_chk@plt>
  4019c2:	4c 89 f7             	mov    %r14,%rdi
  4019c5:	e8 b6 f6 ff ff       	call   401080 <strlen@plt>
  4019ca:	48 c1 e0 20          	shl    $0x20,%rax
  4019ce:	48 ba 00 00 00 00 ff 	movabs $0x3ff00000000,%rdx
  4019d5:	03 00 00 
  4019d8:	48 29 c2             	sub    %rax,%rdx
  4019db:	48 c1 fa 20          	sar    $0x20,%rdx
  4019df:	b9 00 04 00 00       	mov    $0x400,%ecx
  4019e4:	4c 89 f7             	mov    %r14,%rdi
  4019e7:	4c 89 e6             	mov    %r12,%rsi
  4019ea:	e8 91 f7 ff ff       	call   401180 <__strncat_chk@plt>
  4019ef:	0f b6 5c 24 30       	movzbl 0x30(%rsp),%ebx
  4019f4:	bf 01 00 00 00       	mov    $0x1,%edi
  4019f9:	31 f6                	xor    %esi,%esi
  4019fb:	31 d2                	xor    %edx,%edx
  4019fd:	31 c0                	xor    %eax,%eax
  4019ff:	e8 4c 1c 00 00       	call   403650 <__libirc_print>
  401a04:	84 db                	test   %bl,%bl
  401a06:	75 15                	jne    401a1d <__intel_new_feature_proc_init_n+0x31d>
  401a08:	bf 01 00 00 00       	mov    $0x1,%edi
  401a0d:	be 3a 00 00 00       	mov    $0x3a,%esi
  401a12:	31 d2                	xor    %edx,%edx
  401a14:	31 c0                	xor    %eax,%eax
  401a16:	e8 35 1c 00 00       	call   403650 <__libirc_print>
  401a1b:	eb 1b                	jmp    401a38 <__intel_new_feature_proc_init_n+0x338>
  401a1d:	48 8d 4c 24 30       	lea    0x30(%rsp),%rcx
  401a22:	bf 01 00 00 00       	mov    $0x1,%edi
  401a27:	be 38 00 00 00       	mov    $0x38,%esi
  401a2c:	ba 01 00 00 00       	mov    $0x1,%edx
  401a31:	31 c0                	xor    %eax,%eax
  401a33:	e8 18 1c 00 00       	call   403650 <__libirc_print>
  401a38:	bf 01 00 00 00       	mov    $0x1,%edi
  401a3d:	31 f6                	xor    %esi,%esi
  401a3f:	31 d2                	xor    %edx,%edx
  401a41:	31 c0                	xor    %eax,%eax
  401a43:	e8 08 1c 00 00       	call   403650 <__libirc_print>
  401a48:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  401a4f:	00 00 
  401a51:	48 3b 84 24 30 04 00 	cmp    0x430(%rsp),%rax
  401a58:	00 
  401a59:	75 0a                	jne    401a65 <__intel_new_feature_proc_init_n+0x365>
  401a5b:	bf 01 00 00 00       	mov    $0x1,%edi
  401a60:	e8 fb f6 ff ff       	call   401160 <exit@plt>
  401a65:	e8 26 f6 ff ff       	call   401090 <__stack_chk_fail@plt>
  401a6a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401a70 <__intel_new_feature_proc_init>:
  401a70:	f3 0f 1e fa          	endbr64
  401a74:	53                   	push   %rbx
  401a75:	89 fb                	mov    %edi,%ebx
  401a77:	31 ff                	xor    %edi,%edi
  401a79:	e8 82 fc ff ff       	call   401700 <__intel_new_feature_proc_init_n>
  401a7e:	48 c7 c7 d0 70 40 00 	mov    $0x4070d0,%rdi
  401a85:	be 06 00 00 00       	mov    $0x6,%esi
  401a8a:	e8 61 17 00 00       	call   4031f0 <__libirc_get_cpu_feature>
  401a8f:	83 f8 01             	cmp    $0x1,%eax
  401a92:	75 0a                	jne    401a9e <__intel_new_feature_proc_init+0x2e>
  401a94:	31 ff                	xor    %edi,%edi
  401a96:	89 de                	mov    %ebx,%esi
  401a98:	5b                   	pop    %rbx
  401a99:	e9 52 00 00 00       	jmp    401af0 <__intel_proc_init_ftzdazule>
  401a9e:	85 c0                	test   %eax,%eax
  401aa0:	78 02                	js     401aa4 <__intel_new_feature_proc_init+0x34>
  401aa2:	5b                   	pop    %rbx
  401aa3:	c3                   	ret
  401aa4:	bf 01 00 00 00       	mov    $0x1,%edi
  401aa9:	31 f6                	xor    %esi,%esi
  401aab:	31 d2                	xor    %edx,%edx
  401aad:	31 c0                	xor    %eax,%eax
  401aaf:	e8 9c 1b 00 00       	call   403650 <__libirc_print>
  401ab4:	bf 01 00 00 00       	mov    $0x1,%edi
  401ab9:	be 3a 00 00 00       	mov    $0x3a,%esi
  401abe:	31 d2                	xor    %edx,%edx
  401ac0:	31 c0                	xor    %eax,%eax
  401ac2:	e8 89 1b 00 00       	call   403650 <__libirc_print>
  401ac7:	bf 01 00 00 00       	mov    $0x1,%edi
  401acc:	31 f6                	xor    %esi,%esi
  401ace:	31 d2                	xor    %edx,%edx
  401ad0:	31 c0                	xor    %eax,%eax
  401ad2:	e8 79 1b 00 00       	call   403650 <__libirc_print>
  401ad7:	bf 01 00 00 00       	mov    $0x1,%edi
  401adc:	e8 7f f6 ff ff       	call   401160 <exit@plt>
  401ae1:	0f 1f 00             	nopl   (%rax)
  401ae4:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  401aeb:	00 00 00 
  401aee:	66 90                	xchg   %ax,%ax

0000000000401af0 <__intel_proc_init_ftzdazule>:
  401af0:	f3 0f 1e fa          	endbr64
  401af4:	55                   	push   %rbp
  401af5:	41 56                	push   %r14
  401af7:	53                   	push   %rbx
  401af8:	48 81 ec 20 02 00 00 	sub    $0x220,%rsp
  401aff:	89 f3                	mov    %esi,%ebx
  401b01:	41 89 f6             	mov    %esi,%r14d
  401b04:	41 83 e6 04          	and    $0x4,%r14d
  401b08:	89 f5                	mov    %esi,%ebp
  401b0a:	83 e5 02             	and    $0x2,%ebp
  401b0d:	74 07                	je     401b16 <__intel_proc_init_ftzdazule+0x26>
  401b0f:	89 f8                	mov    %edi,%eax
  401b11:	83 e0 02             	and    $0x2,%eax
  401b14:	74 12                	je     401b28 <__intel_proc_init_ftzdazule+0x38>
  401b16:	31 c0                	xor    %eax,%eax
  401b18:	45 85 f6             	test   %r14d,%r14d
  401b1b:	74 38                	je     401b55 <__intel_proc_init_ftzdazule+0x65>
  401b1d:	b8 01 00 00 00       	mov    $0x1,%eax
  401b22:	40 f6 c7 04          	test   $0x4,%dil
  401b26:	75 2d                	jne    401b55 <__intel_proc_init_ftzdazule+0x65>
  401b28:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
  401b2d:	ba 00 02 00 00       	mov    $0x200,%edx
  401b32:	31 f6                	xor    %esi,%esi
  401b34:	e8 77 f5 ff ff       	call   4010b0 <memset@plt>
  401b39:	0f ae 44 24 20       	fxsave 0x20(%rsp)
  401b3e:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
  401b42:	89 c1                	mov    %eax,%ecx
  401b44:	c1 e1 19             	shl    $0x19,%ecx
  401b47:	c1 f9 1f             	sar    $0x1f,%ecx
  401b4a:	21 cd                	and    %ecx,%ebp
  401b4c:	c1 e0 0e             	shl    $0xe,%eax
  401b4f:	c1 f8 1f             	sar    $0x1f,%eax
  401b52:	44 21 f0             	and    %r14d,%eax
  401b55:	f6 c3 01             	test   $0x1,%bl
  401b58:	74 17                	je     401b71 <__intel_proc_init_ftzdazule+0x81>
  401b5a:	0f ae 5c 24 1c       	stmxcsr 0x1c(%rsp)
  401b5f:	b9 00 80 00 00       	mov    $0x8000,%ecx
  401b64:	0b 4c 24 1c          	or     0x1c(%rsp),%ecx
  401b68:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
  401b6c:	0f ae 54 24 18       	ldmxcsr 0x18(%rsp)
  401b71:	85 ed                	test   %ebp,%ebp
  401b73:	74 15                	je     401b8a <__intel_proc_init_ftzdazule+0x9a>
  401b75:	0f ae 5c 24 14       	stmxcsr 0x14(%rsp)
  401b7a:	8b 4c 24 14          	mov    0x14(%rsp),%ecx
  401b7e:	83 c9 40             	or     $0x40,%ecx
  401b81:	89 4c 24 10          	mov    %ecx,0x10(%rsp)
  401b85:	0f ae 54 24 10       	ldmxcsr 0x10(%rsp)
  401b8a:	85 c0                	test   %eax,%eax
  401b8c:	74 17                	je     401ba5 <__intel_proc_init_ftzdazule+0xb5>
  401b8e:	0f ae 5c 24 0c       	stmxcsr 0xc(%rsp)
  401b93:	b8 00 00 02 00       	mov    $0x20000,%eax
  401b98:	0b 44 24 0c          	or     0xc(%rsp),%eax
  401b9c:	89 44 24 08          	mov    %eax,0x8(%rsp)
  401ba0:	0f ae 54 24 08       	ldmxcsr 0x8(%rsp)
  401ba5:	48 81 c4 20 02 00 00 	add    $0x220,%rsp
  401bac:	5b                   	pop    %rbx
  401bad:	41 5e                	pop    %r14
  401baf:	5d                   	pop    %rbp
  401bb0:	c3                   	ret
  401bb1:	0f 1f 00             	nopl   (%rax)
  401bb4:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  401bbb:	00 00 00 
  401bbe:	66 90                	xchg   %ax,%ax

0000000000401bc0 <__intel_cpu_features_init>:
  401bc0:	f3 0f 1e fa          	endbr64
  401bc4:	50                   	push   %rax
  401bc5:	b8 01 00 00 00       	mov    $0x1,%eax
  401bca:	e8 11 00 00 00       	call   401be0 <__intel_cpu_features_init_body>
  401bcf:	48 83 c4 08          	add    $0x8,%rsp
  401bd3:	c3                   	ret
  401bd4:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  401bdb:	00 00 00 
  401bde:	66 90                	xchg   %ax,%ax

0000000000401be0 <__intel_cpu_features_init_body>:
  401be0:	41 53                	push   %r11
  401be2:	41 52                	push   %r10
  401be4:	41 51                	push   %r9
  401be6:	41 50                	push   %r8
  401be8:	52                   	push   %rdx
  401be9:	51                   	push   %rcx
  401bea:	56                   	push   %rsi
  401beb:	57                   	push   %rdi
  401bec:	55                   	push   %rbp
  401bed:	53                   	push   %rbx
  401bee:	48 81 ec 38 01 00 00 	sub    $0x138,%rsp
  401bf5:	44 0f 29 bc 24 20 01 	movaps %xmm15,0x120(%rsp)
  401bfc:	00 00 
  401bfe:	44 0f 29 b4 24 10 01 	movaps %xmm14,0x110(%rsp)
  401c05:	00 00 
  401c07:	44 0f 29 ac 24 00 01 	movaps %xmm13,0x100(%rsp)
  401c0e:	00 00 
  401c10:	44 0f 29 a4 24 f0 00 	movaps %xmm12,0xf0(%rsp)
  401c17:	00 00 
  401c19:	44 0f 29 9c 24 e0 00 	movaps %xmm11,0xe0(%rsp)
  401c20:	00 00 
  401c22:	44 0f 29 94 24 d0 00 	movaps %xmm10,0xd0(%rsp)
  401c29:	00 00 
  401c2b:	44 0f 29 8c 24 c0 00 	movaps %xmm9,0xc0(%rsp)
  401c32:	00 00 
  401c34:	44 0f 29 84 24 b0 00 	movaps %xmm8,0xb0(%rsp)
  401c3b:	00 00 
  401c3d:	0f 29 bc 24 a0 00 00 	movaps %xmm7,0xa0(%rsp)
  401c44:	00 
  401c45:	0f 29 b4 24 90 00 00 	movaps %xmm6,0x90(%rsp)
  401c4c:	00 
  401c4d:	0f 29 ac 24 80 00 00 	movaps %xmm5,0x80(%rsp)
  401c54:	00 
  401c55:	0f 29 64 24 70       	movaps %xmm4,0x70(%rsp)
  401c5a:	0f 29 5c 24 60       	movaps %xmm3,0x60(%rsp)
  401c5f:	0f 29 54 24 50       	movaps %xmm2,0x50(%rsp)
  401c64:	0f 29 4c 24 40       	movaps %xmm1,0x40(%rsp)
  401c69:	0f 29 44 24 30       	movaps %xmm0,0x30(%rsp)
  401c6e:	89 c5                	mov    %eax,%ebp
  401c70:	0f 57 c0             	xorps  %xmm0,%xmm0
  401c73:	0f 29 04 24          	movaps %xmm0,(%rsp)
  401c77:	0f 29 44 24 20       	movaps %xmm0,0x20(%rsp)
  401c7c:	48 89 e0             	mov    %rsp,%rax
  401c7f:	b9 01 00 00 00       	mov    $0x1,%ecx
  401c84:	e8 b7 15 00 00       	call   403240 <__libirc_set_cpu_feature>
  401c89:	85 c0                	test   %eax,%eax
  401c8b:	0f 85 81 03 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401c91:	31 c0                	xor    %eax,%eax
  401c93:	0f a2                	cpuid
  401c95:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  401c99:	89 5c 24 18          	mov    %ebx,0x18(%rsp)
  401c9d:	89 4c 24 14          	mov    %ecx,0x14(%rsp)
  401ca1:	89 54 24 10          	mov    %edx,0x10(%rsp)
  401ca5:	83 7c 24 1c 00       	cmpl   $0x0,0x1c(%rsp)
  401caa:	0f 84 55 03 00 00    	je     402005 <__intel_cpu_features_init_body+0x425>
  401cb0:	83 fd 01             	cmp    $0x1,%ebp
  401cb3:	75 2a                	jne    401cdf <__intel_cpu_features_init_body+0xff>
  401cb5:	81 7c 24 18 47 65 6e 	cmpl   $0x756e6547,0x18(%rsp)
  401cbc:	75 
  401cbd:	0f 85 42 03 00 00    	jne    402005 <__intel_cpu_features_init_body+0x425>
  401cc3:	81 7c 24 10 69 6e 65 	cmpl   $0x49656e69,0x10(%rsp)
  401cca:	49 
  401ccb:	0f 85 34 03 00 00    	jne    402005 <__intel_cpu_features_init_body+0x425>
  401cd1:	81 7c 24 14 6e 74 65 	cmpl   $0x6c65746e,0x14(%rsp)
  401cd8:	6c 
  401cd9:	0f 85 26 03 00 00    	jne    402005 <__intel_cpu_features_init_body+0x425>
  401cdf:	b8 01 00 00 00       	mov    $0x1,%eax
  401ce4:	0f a2                	cpuid
  401ce6:	41 89 d2             	mov    %edx,%r10d
  401ce9:	41 89 c8             	mov    %ecx,%r8d
  401cec:	41 f6 c2 01          	test   $0x1,%r10b
  401cf0:	74 15                	je     401d07 <__intel_cpu_features_init_body+0x127>
  401cf2:	48 89 e0             	mov    %rsp,%rax
  401cf5:	b9 02 00 00 00       	mov    $0x2,%ecx
  401cfa:	e8 41 15 00 00       	call   403240 <__libirc_set_cpu_feature>
  401cff:	85 c0                	test   %eax,%eax
  401d01:	0f 85 0b 03 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401d07:	66 45 85 d2          	test   %r10w,%r10w
  401d0b:	79 15                	jns    401d22 <__intel_cpu_features_init_body+0x142>
  401d0d:	48 89 e0             	mov    %rsp,%rax
  401d10:	b9 03 00 00 00       	mov    $0x3,%ecx
  401d15:	e8 26 15 00 00       	call   403240 <__libirc_set_cpu_feature>
  401d1a:	85 c0                	test   %eax,%eax
  401d1c:	0f 85 f0 02 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401d22:	41 f7 c2 00 00 80 00 	test   $0x800000,%r10d
  401d29:	74 15                	je     401d40 <__intel_cpu_features_init_body+0x160>
  401d2b:	48 89 e0             	mov    %rsp,%rax
  401d2e:	b9 04 00 00 00       	mov    $0x4,%ecx
  401d33:	e8 08 15 00 00       	call   403240 <__libirc_set_cpu_feature>
  401d38:	85 c0                	test   %eax,%eax
  401d3a:	0f 85 d2 02 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401d40:	41 f7 c2 00 00 00 01 	test   $0x1000000,%r10d
  401d47:	0f 85 54 03 00 00    	jne    4020a1 <__intel_cpu_features_init_body+0x4c1>
  401d4d:	41 f7 c0 00 00 00 40 	test   $0x40000000,%r8d
  401d54:	74 15                	je     401d6b <__intel_cpu_features_init_body+0x18b>
  401d56:	48 89 e0             	mov    %rsp,%rax
  401d59:	b9 12 00 00 00       	mov    $0x12,%ecx
  401d5e:	e8 dd 14 00 00       	call   403240 <__libirc_set_cpu_feature>
  401d63:	85 c0                	test   %eax,%eax
  401d65:	0f 85 a7 02 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401d6b:	41 f7 c2 00 00 00 01 	test   $0x1000000,%r10d
  401d72:	75 10                	jne    401d84 <__intel_cpu_features_init_body+0x1a4>
  401d74:	b8 07 00 00 00       	mov    $0x7,%eax
  401d79:	31 c9                	xor    %ecx,%ecx
  401d7b:	0f a2                	cpuid
  401d7d:	89 cf                	mov    %ecx,%edi
  401d7f:	89 d6                	mov    %edx,%esi
  401d81:	41 89 d9             	mov    %ebx,%r9d
  401d84:	44 89 c8             	mov    %r9d,%eax
  401d87:	f7 d0                	not    %eax
  401d89:	a9 08 01 00 00       	test   $0x108,%eax
  401d8e:	75 15                	jne    401da5 <__intel_cpu_features_init_body+0x1c5>
  401d90:	48 89 e0             	mov    %rsp,%rax
  401d93:	b9 14 00 00 00       	mov    $0x14,%ecx
  401d98:	e8 a3 14 00 00       	call   403240 <__libirc_set_cpu_feature>
  401d9d:	85 c0                	test   %eax,%eax
  401d9f:	0f 85 6d 02 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401da5:	41 f6 c1 04          	test   $0x4,%r9b
  401da9:	74 15                	je     401dc0 <__intel_cpu_features_init_body+0x1e0>
  401dab:	48 89 e0             	mov    %rsp,%rax
  401dae:	b9 36 00 00 00       	mov    $0x36,%ecx
  401db3:	e8 88 14 00 00       	call   403240 <__libirc_set_cpu_feature>
  401db8:	85 c0                	test   %eax,%eax
  401dba:	0f 85 52 02 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401dc0:	41 f6 c1 10          	test   $0x10,%r9b
  401dc4:	74 15                	je     401ddb <__intel_cpu_features_init_body+0x1fb>
  401dc6:	48 89 e0             	mov    %rsp,%rax
  401dc9:	b9 16 00 00 00       	mov    $0x16,%ecx
  401dce:	e8 6d 14 00 00       	call   403240 <__libirc_set_cpu_feature>
  401dd3:	85 c0                	test   %eax,%eax
  401dd5:	0f 85 37 02 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401ddb:	41 f7 c1 00 08 00 00 	test   $0x800,%r9d
  401de2:	74 15                	je     401df9 <__intel_cpu_features_init_body+0x219>
  401de4:	48 89 e0             	mov    %rsp,%rax
  401de7:	b9 17 00 00 00       	mov    $0x17,%ecx
  401dec:	e8 4f 14 00 00       	call   403240 <__libirc_set_cpu_feature>
  401df1:	85 c0                	test   %eax,%eax
  401df3:	0f 85 19 02 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401df9:	41 f7 c1 00 00 08 00 	test   $0x80000,%r9d
  401e00:	74 15                	je     401e17 <__intel_cpu_features_init_body+0x237>
  401e02:	48 89 e0             	mov    %rsp,%rax
  401e05:	b9 1d 00 00 00       	mov    $0x1d,%ecx
  401e0a:	e8 31 14 00 00       	call   403240 <__libirc_set_cpu_feature>
  401e0f:	85 c0                	test   %eax,%eax
  401e11:	0f 85 fb 01 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401e17:	41 f7 c1 00 00 04 00 	test   $0x40000,%r9d
  401e1e:	74 15                	je     401e35 <__intel_cpu_features_init_body+0x255>
  401e20:	48 89 e0             	mov    %rsp,%rax
  401e23:	b9 1e 00 00 00       	mov    $0x1e,%ecx
  401e28:	e8 13 14 00 00       	call   403240 <__libirc_set_cpu_feature>
  401e2d:	85 c0                	test   %eax,%eax
  401e2f:	0f 85 dd 01 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401e35:	41 f7 c1 00 00 00 01 	test   $0x1000000,%r9d
  401e3c:	74 15                	je     401e53 <__intel_cpu_features_init_body+0x273>
  401e3e:	48 89 e0             	mov    %rsp,%rax
  401e41:	b9 32 00 00 00       	mov    $0x32,%ecx
  401e46:	e8 f5 13 00 00       	call   403240 <__libirc_set_cpu_feature>
  401e4b:	85 c0                	test   %eax,%eax
  401e4d:	0f 85 bf 01 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401e53:	b8 01 00 00 80       	mov    $0x80000001,%eax
  401e58:	0f a2                	cpuid
  401e5a:	f6 c1 20             	test   $0x20,%cl
  401e5d:	74 15                	je     401e74 <__intel_cpu_features_init_body+0x294>
  401e5f:	48 89 e0             	mov    %rsp,%rax
  401e62:	b9 15 00 00 00       	mov    $0x15,%ecx
  401e67:	e8 d4 13 00 00       	call   403240 <__libirc_set_cpu_feature>
  401e6c:	85 c0                	test   %eax,%eax
  401e6e:	0f 85 9e 01 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401e74:	b8 08 00 00 80       	mov    $0x80000008,%eax
  401e79:	0f a2                	cpuid
  401e7b:	f7 c3 00 02 00 00    	test   $0x200,%ebx
  401e81:	74 15                	je     401e98 <__intel_cpu_features_init_body+0x2b8>
  401e83:	48 89 e0             	mov    %rsp,%rax
  401e86:	b9 37 00 00 00       	mov    $0x37,%ecx
  401e8b:	e8 b0 13 00 00       	call   403240 <__libirc_set_cpu_feature>
  401e90:	85 c0                	test   %eax,%eax
  401e92:	0f 85 7a 01 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401e98:	40 f6 c7 20          	test   $0x20,%dil
  401e9c:	74 15                	je     401eb3 <__intel_cpu_features_init_body+0x2d3>
  401e9e:	48 89 e0             	mov    %rsp,%rax
  401ea1:	b9 3e 00 00 00       	mov    $0x3e,%ecx
  401ea6:	e8 95 13 00 00       	call   403240 <__libirc_set_cpu_feature>
  401eab:	85 c0                	test   %eax,%eax
  401ead:	0f 85 5f 01 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401eb3:	40 84 ff             	test   %dil,%dil
  401eb6:	79 15                	jns    401ecd <__intel_cpu_features_init_body+0x2ed>
  401eb8:	48 89 e0             	mov    %rsp,%rax
  401ebb:	b9 35 00 00 00       	mov    $0x35,%ecx
  401ec0:	e8 7b 13 00 00       	call   403240 <__libirc_set_cpu_feature>
  401ec5:	85 c0                	test   %eax,%eax
  401ec7:	0f 85 45 01 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401ecd:	f7 c7 00 01 00 00    	test   $0x100,%edi
  401ed3:	74 15                	je     401eea <__intel_cpu_features_init_body+0x30a>
  401ed5:	48 89 e0             	mov    %rsp,%rax
  401ed8:	b9 2e 00 00 00       	mov    $0x2e,%ecx
  401edd:	e8 5e 13 00 00       	call   403240 <__libirc_set_cpu_feature>
  401ee2:	85 c0                	test   %eax,%eax
  401ee4:	0f 85 28 01 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401eea:	f7 c7 00 00 40 00    	test   $0x400000,%edi
  401ef0:	74 15                	je     401f07 <__intel_cpu_features_init_body+0x327>
  401ef2:	48 89 e0             	mov    %rsp,%rax
  401ef5:	b9 33 00 00 00       	mov    $0x33,%ecx
  401efa:	e8 41 13 00 00       	call   403240 <__libirc_set_cpu_feature>
  401eff:	85 c0                	test   %eax,%eax
  401f01:	0f 85 0b 01 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401f07:	f7 c7 00 00 00 01    	test   $0x1000000,%edi
  401f0d:	74 15                	je     401f24 <__intel_cpu_features_init_body+0x344>
  401f0f:	48 89 e0             	mov    %rsp,%rax
  401f12:	b9 3b 00 00 00       	mov    $0x3b,%ecx
  401f17:	e8 24 13 00 00       	call   403240 <__libirc_set_cpu_feature>
  401f1c:	85 c0                	test   %eax,%eax
  401f1e:	0f 85 ee 00 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401f24:	f7 c7 00 00 00 08    	test   $0x8000000,%edi
  401f2a:	74 15                	je     401f41 <__intel_cpu_features_init_body+0x361>
  401f2c:	48 89 e0             	mov    %rsp,%rax
  401f2f:	b9 3c 00 00 00       	mov    $0x3c,%ecx
  401f34:	e8 07 13 00 00       	call   403240 <__libirc_set_cpu_feature>
  401f39:	85 c0                	test   %eax,%eax
  401f3b:	0f 85 d1 00 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401f41:	f7 c7 00 00 00 10    	test   $0x10000000,%edi
  401f47:	74 15                	je     401f5e <__intel_cpu_features_init_body+0x37e>
  401f49:	48 89 e0             	mov    %rsp,%rax
  401f4c:	b9 3d 00 00 00       	mov    $0x3d,%ecx
  401f51:	e8 ea 12 00 00       	call   403240 <__libirc_set_cpu_feature>
  401f56:	85 c0                	test   %eax,%eax
  401f58:	0f 85 b4 00 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401f5e:	f7 c7 00 00 00 20    	test   $0x20000000,%edi
  401f64:	74 15                	je     401f7b <__intel_cpu_features_init_body+0x39b>
  401f66:	48 89 e0             	mov    %rsp,%rax
  401f69:	b9 40 00 00 00       	mov    $0x40,%ecx
  401f6e:	e8 cd 12 00 00       	call   403240 <__libirc_set_cpu_feature>
  401f73:	85 c0                	test   %eax,%eax
  401f75:	0f 85 97 00 00 00    	jne    402012 <__intel_cpu_features_init_body+0x432>
  401f7b:	f7 c6 00 00 10 00    	test   $0x100000,%esi
  401f81:	74 11                	je     401f94 <__intel_cpu_features_init_body+0x3b4>
  401f83:	48 89 e0             	mov    %rsp,%rax
  401f86:	b9 34 00 00 00       	mov    $0x34,%ecx
  401f8b:	e8 b0 12 00 00       	call   403240 <__libirc_set_cpu_feature>
  401f90:	85 c0                	test   %eax,%eax
  401f92:	75 7e                	jne    402012 <__intel_cpu_features_init_body+0x432>
  401f94:	f7 c6 00 00 04 00    	test   $0x40000,%esi
  401f9a:	74 11                	je     401fad <__intel_cpu_features_init_body+0x3cd>
  401f9c:	48 89 e0             	mov    %rsp,%rax
  401f9f:	b9 38 00 00 00       	mov    $0x38,%ecx
  401fa4:	e8 97 12 00 00       	call   403240 <__libirc_set_cpu_feature>
  401fa9:	85 c0                	test   %eax,%eax
  401fab:	75 65                	jne    402012 <__intel_cpu_features_init_body+0x432>
  401fad:	b8 14 00 00 00       	mov    $0x14,%eax
  401fb2:	31 c9                	xor    %ecx,%ecx
  401fb4:	0f a2                	cpuid
  401fb6:	f6 c3 10             	test   $0x10,%bl
  401fb9:	74 11                	je     401fcc <__intel_cpu_features_init_body+0x3ec>
  401fbb:	48 89 e0             	mov    %rsp,%rax
  401fbe:	b9 1b 00 00 00       	mov    $0x1b,%ecx
  401fc3:	e8 78 12 00 00       	call   403240 <__libirc_set_cpu_feature>
  401fc8:	85 c0                	test   %eax,%eax
  401fca:	75 46                	jne    402012 <__intel_cpu_features_init_body+0x432>
  401fcc:	f7 c7 00 00 80 00    	test   $0x800000,%edi
  401fd2:	0f 85 3a 02 00 00    	jne    402212 <__intel_cpu_features_init_body+0x632>
  401fd8:	41 f7 c0 00 00 00 08 	test   $0x8000000,%r8d
  401fdf:	0f 85 88 02 00 00    	jne    40226d <__intel_cpu_features_init_body+0x68d>
  401fe5:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
  401fea:	e8 b1 12 00 00       	call   4032a0 <__libirc_handle_intel_isa_disable>
  401fef:	85 c0                	test   %eax,%eax
  401ff1:	0f 8e 09 06 00 00    	jle    402600 <__intel_cpu_features_init_body+0xa20>
  401ff7:	0f 28 44 24 20       	movaps 0x20(%rsp),%xmm0
  401ffc:	0f 55 04 24          	andnps (%rsp),%xmm0
  402000:	e9 ff 05 00 00       	jmp    402604 <__intel_cpu_features_init_body+0xa24>
  402005:	0f 28 04 24          	movaps (%rsp),%xmm0
  402009:	0f 29 05 c0 50 00 00 	movaps %xmm0,0x50c0(%rip)        # 4070d0 <__intel_cpu_feature_indicator>
  402010:	31 c0                	xor    %eax,%eax
  402012:	0f 28 44 24 30       	movaps 0x30(%rsp),%xmm0
  402017:	0f 28 4c 24 40       	movaps 0x40(%rsp),%xmm1
  40201c:	0f 28 54 24 50       	movaps 0x50(%rsp),%xmm2
  402021:	0f 28 5c 24 60       	movaps 0x60(%rsp),%xmm3
  402026:	0f 28 64 24 70       	movaps 0x70(%rsp),%xmm4
  40202b:	0f 28 ac 24 80 00 00 	movaps 0x80(%rsp),%xmm5
  402032:	00 
  402033:	0f 28 b4 24 90 00 00 	movaps 0x90(%rsp),%xmm6
  40203a:	00 
  40203b:	0f 28 bc 24 a0 00 00 	movaps 0xa0(%rsp),%xmm7
  402042:	00 
  402043:	44 0f 28 84 24 b0 00 	movaps 0xb0(%rsp),%xmm8
  40204a:	00 00 
  40204c:	44 0f 28 8c 24 c0 00 	movaps 0xc0(%rsp),%xmm9
  402053:	00 00 
  402055:	44 0f 28 94 24 d0 00 	movaps 0xd0(%rsp),%xmm10
  40205c:	00 00 
  40205e:	44 0f 28 9c 24 e0 00 	movaps 0xe0(%rsp),%xmm11
  402065:	00 00 
  402067:	44 0f 28 a4 24 f0 00 	movaps 0xf0(%rsp),%xmm12
  40206e:	00 00 
  402070:	44 0f 28 ac 24 00 01 	movaps 0x100(%rsp),%xmm13
  402077:	00 00 
  402079:	44 0f 28 b4 24 10 01 	movaps 0x110(%rsp),%xmm14
  402080:	00 00 
  402082:	44 0f 28 bc 24 20 01 	movaps 0x120(%rsp),%xmm15
  402089:	00 00 
  40208b:	48 81 c4 38 01 00 00 	add    $0x138,%rsp
  402092:	5b                   	pop    %rbx
  402093:	5d                   	pop    %rbp
  402094:	5f                   	pop    %rdi
  402095:	5e                   	pop    %rsi
  402096:	59                   	pop    %rcx
  402097:	5a                   	pop    %rdx
  402098:	41 58                	pop    %r8
  40209a:	41 59                	pop    %r9
  40209c:	41 5a                	pop    %r10
  40209e:	41 5b                	pop    %r11
  4020a0:	c3                   	ret
  4020a1:	48 89 e0             	mov    %rsp,%rax
  4020a4:	b9 05 00 00 00       	mov    $0x5,%ecx
  4020a9:	e8 92 11 00 00       	call   403240 <__libirc_set_cpu_feature>
  4020ae:	85 c0                	test   %eax,%eax
  4020b0:	0f 85 5c ff ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4020b6:	41 f7 c2 00 00 00 02 	test   $0x2000000,%r10d
  4020bd:	74 15                	je     4020d4 <__intel_cpu_features_init_body+0x4f4>
  4020bf:	48 89 e0             	mov    %rsp,%rax
  4020c2:	b9 06 00 00 00       	mov    $0x6,%ecx
  4020c7:	e8 74 11 00 00       	call   403240 <__libirc_set_cpu_feature>
  4020cc:	85 c0                	test   %eax,%eax
  4020ce:	0f 85 3e ff ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4020d4:	41 f7 c2 00 00 00 04 	test   $0x4000000,%r10d
  4020db:	74 15                	je     4020f2 <__intel_cpu_features_init_body+0x512>
  4020dd:	48 89 e0             	mov    %rsp,%rax
  4020e0:	b9 07 00 00 00       	mov    $0x7,%ecx
  4020e5:	e8 56 11 00 00       	call   403240 <__libirc_set_cpu_feature>
  4020ea:	85 c0                	test   %eax,%eax
  4020ec:	0f 85 20 ff ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4020f2:	41 f6 c0 01          	test   $0x1,%r8b
  4020f6:	74 15                	je     40210d <__intel_cpu_features_init_body+0x52d>
  4020f8:	48 89 e0             	mov    %rsp,%rax
  4020fb:	b9 08 00 00 00       	mov    $0x8,%ecx
  402100:	e8 3b 11 00 00       	call   403240 <__libirc_set_cpu_feature>
  402105:	85 c0                	test   %eax,%eax
  402107:	0f 85 05 ff ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  40210d:	41 f7 c0 00 02 00 00 	test   $0x200,%r8d
  402114:	74 15                	je     40212b <__intel_cpu_features_init_body+0x54b>
  402116:	48 89 e0             	mov    %rsp,%rax
  402119:	b9 09 00 00 00       	mov    $0x9,%ecx
  40211e:	e8 1d 11 00 00       	call   403240 <__libirc_set_cpu_feature>
  402123:	85 c0                	test   %eax,%eax
  402125:	0f 85 e7 fe ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  40212b:	41 f7 c0 00 00 40 00 	test   $0x400000,%r8d
  402132:	74 15                	je     402149 <__intel_cpu_features_init_body+0x569>
  402134:	48 89 e0             	mov    %rsp,%rax
  402137:	b9 0c 00 00 00       	mov    $0xc,%ecx
  40213c:	e8 ff 10 00 00       	call   403240 <__libirc_set_cpu_feature>
  402141:	85 c0                	test   %eax,%eax
  402143:	0f 85 c9 fe ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402149:	41 f7 c0 00 00 08 00 	test   $0x80000,%r8d
  402150:	74 15                	je     402167 <__intel_cpu_features_init_body+0x587>
  402152:	48 89 e0             	mov    %rsp,%rax
  402155:	b9 0a 00 00 00       	mov    $0xa,%ecx
  40215a:	e8 e1 10 00 00       	call   403240 <__libirc_set_cpu_feature>
  40215f:	85 c0                	test   %eax,%eax
  402161:	0f 85 ab fe ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402167:	41 f7 c0 00 00 10 00 	test   $0x100000,%r8d
  40216e:	74 15                	je     402185 <__intel_cpu_features_init_body+0x5a5>
  402170:	48 89 e0             	mov    %rsp,%rax
  402173:	b9 0b 00 00 00       	mov    $0xb,%ecx
  402178:	e8 c3 10 00 00       	call   403240 <__libirc_set_cpu_feature>
  40217d:	85 c0                	test   %eax,%eax
  40217f:	0f 85 8d fe ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402185:	41 f7 c0 00 00 80 00 	test   $0x800000,%r8d
  40218c:	74 15                	je     4021a3 <__intel_cpu_features_init_body+0x5c3>
  40218e:	48 89 e0             	mov    %rsp,%rax
  402191:	b9 0d 00 00 00       	mov    $0xd,%ecx
  402196:	e8 a5 10 00 00       	call   403240 <__libirc_set_cpu_feature>
  40219b:	85 c0                	test   %eax,%eax
  40219d:	0f 85 6f fe ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4021a3:	41 f6 c0 02          	test   $0x2,%r8b
  4021a7:	74 15                	je     4021be <__intel_cpu_features_init_body+0x5de>
  4021a9:	48 89 e0             	mov    %rsp,%rax
  4021ac:	b9 0e 00 00 00       	mov    $0xe,%ecx
  4021b1:	e8 8a 10 00 00       	call   403240 <__libirc_set_cpu_feature>
  4021b6:	85 c0                	test   %eax,%eax
  4021b8:	0f 85 54 fe ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4021be:	41 f7 c0 00 00 00 02 	test   $0x2000000,%r8d
  4021c5:	74 15                	je     4021dc <__intel_cpu_features_init_body+0x5fc>
  4021c7:	48 89 e0             	mov    %rsp,%rax
  4021ca:	b9 0f 00 00 00       	mov    $0xf,%ecx
  4021cf:	e8 6c 10 00 00       	call   403240 <__libirc_set_cpu_feature>
  4021d4:	85 c0                	test   %eax,%eax
  4021d6:	0f 85 36 fe ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4021dc:	b8 07 00 00 00       	mov    $0x7,%eax
  4021e1:	31 c9                	xor    %ecx,%ecx
  4021e3:	0f a2                	cpuid
  4021e5:	89 cf                	mov    %ecx,%edi
  4021e7:	89 d6                	mov    %edx,%esi
  4021e9:	41 89 d9             	mov    %ebx,%r9d
  4021ec:	f7 c3 00 00 00 20    	test   $0x20000000,%ebx
  4021f2:	0f 84 55 fb ff ff    	je     401d4d <__intel_cpu_features_init_body+0x16d>
  4021f8:	48 89 e0             	mov    %rsp,%rax
  4021fb:	b9 24 00 00 00       	mov    $0x24,%ecx
  402200:	e8 3b 10 00 00       	call   403240 <__libirc_set_cpu_feature>
  402205:	85 c0                	test   %eax,%eax
  402207:	0f 85 05 fe ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  40220d:	e9 3b fb ff ff       	jmp    401d4d <__intel_cpu_features_init_body+0x16d>
  402212:	48 89 e0             	mov    %rsp,%rax
  402215:	b9 01 00 00 00       	mov    $0x1,%ecx
  40221a:	e8 21 10 00 00       	call   403240 <__libirc_set_cpu_feature>
  40221f:	85 c0                	test   %eax,%eax
  402221:	0f 85 eb fd ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402227:	b8 19 00 00 00       	mov    $0x19,%eax
  40222c:	31 c9                	xor    %ecx,%ecx
  40222e:	0f a2                	cpuid
  402230:	f6 c3 01             	test   $0x1,%bl
  402233:	74 15                	je     40224a <__intel_cpu_features_init_body+0x66a>
  402235:	48 89 e0             	mov    %rsp,%rax
  402238:	b9 45 00 00 00       	mov    $0x45,%ecx
  40223d:	e8 fe 0f 00 00       	call   403240 <__libirc_set_cpu_feature>
  402242:	85 c0                	test   %eax,%eax
  402244:	0f 85 c8 fd ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  40224a:	f6 c3 04             	test   $0x4,%bl
  40224d:	0f 84 85 fd ff ff    	je     401fd8 <__intel_cpu_features_init_body+0x3f8>
  402253:	48 89 e0             	mov    %rsp,%rax
  402256:	b9 46 00 00 00       	mov    $0x46,%ecx
  40225b:	e8 e0 0f 00 00       	call   403240 <__libirc_set_cpu_feature>
  402260:	85 c0                	test   %eax,%eax
  402262:	0f 85 aa fd ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402268:	e9 6b fd ff ff       	jmp    401fd8 <__intel_cpu_features_init_body+0x3f8>
  40226d:	48 89 e0             	mov    %rsp,%rax
  402270:	b9 01 00 00 00       	mov    $0x1,%ecx
  402275:	e8 c6 0f 00 00       	call   403240 <__libirc_set_cpu_feature>
  40227a:	85 c0                	test   %eax,%eax
  40227c:	0f 85 90 fd ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402282:	31 c9                	xor    %ecx,%ecx
  402284:	0f 01 d0             	xgetbv
  402287:	41 89 c2             	mov    %eax,%r10d
  40228a:	41 f7 d2             	not    %r10d
  40228d:	41 f7 c2 00 00 06 00 	test   $0x60000,%r10d
  402294:	75 6c                	jne    402302 <__intel_cpu_features_init_body+0x722>
  402296:	48 89 e0             	mov    %rsp,%rax
  402299:	b9 01 00 00 00       	mov    $0x1,%ecx
  40229e:	e8 9d 0f 00 00       	call   403240 <__libirc_set_cpu_feature>
  4022a3:	85 c0                	test   %eax,%eax
  4022a5:	0f 85 67 fd ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4022ab:	f7 c6 00 00 00 01    	test   $0x1000000,%esi
  4022b1:	74 15                	je     4022c8 <__intel_cpu_features_init_body+0x6e8>
  4022b3:	48 89 e0             	mov    %rsp,%rax
  4022b6:	b9 42 00 00 00       	mov    $0x42,%ecx
  4022bb:	e8 80 0f 00 00       	call   403240 <__libirc_set_cpu_feature>
  4022c0:	85 c0                	test   %eax,%eax
  4022c2:	0f 85 4a fd ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4022c8:	f7 c6 00 00 00 02    	test   $0x2000000,%esi
  4022ce:	74 15                	je     4022e5 <__intel_cpu_features_init_body+0x705>
  4022d0:	48 89 e0             	mov    %rsp,%rax
  4022d3:	b9 43 00 00 00       	mov    $0x43,%ecx
  4022d8:	e8 63 0f 00 00       	call   403240 <__libirc_set_cpu_feature>
  4022dd:	85 c0                	test   %eax,%eax
  4022df:	0f 85 2d fd ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4022e5:	f7 c6 00 00 40 00    	test   $0x400000,%esi
  4022eb:	74 15                	je     402302 <__intel_cpu_features_init_body+0x722>
  4022ed:	48 89 e0             	mov    %rsp,%rax
  4022f0:	b9 44 00 00 00       	mov    $0x44,%ecx
  4022f5:	e8 46 0f 00 00       	call   403240 <__libirc_set_cpu_feature>
  4022fa:	85 c0                	test   %eax,%eax
  4022fc:	0f 85 10 fd ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402302:	41 f6 c2 06          	test   $0x6,%r10b
  402306:	0f 85 d9 fc ff ff    	jne    401fe5 <__intel_cpu_features_init_body+0x405>
  40230c:	48 89 e0             	mov    %rsp,%rax
  40230f:	b9 01 00 00 00       	mov    $0x1,%ecx
  402314:	e8 27 0f 00 00       	call   403240 <__libirc_set_cpu_feature>
  402319:	85 c0                	test   %eax,%eax
  40231b:	0f 85 f1 fc ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402321:	41 f7 c0 00 00 00 10 	test   $0x10000000,%r8d
  402328:	0f 85 f1 02 00 00    	jne    40261f <__intel_cpu_features_init_body+0xa3f>
  40232e:	41 f7 c0 00 00 00 20 	test   $0x20000000,%r8d
  402335:	74 15                	je     40234c <__intel_cpu_features_init_body+0x76c>
  402337:	48 89 e0             	mov    %rsp,%rax
  40233a:	b9 11 00 00 00       	mov    $0x11,%ecx
  40233f:	e8 fc 0e 00 00       	call   403240 <__libirc_set_cpu_feature>
  402344:	85 c0                	test   %eax,%eax
  402346:	0f 85 c6 fc ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  40234c:	41 f6 c1 20          	test   $0x20,%r9b
  402350:	74 15                	je     402367 <__intel_cpu_features_init_body+0x787>
  402352:	48 89 e0             	mov    %rsp,%rax
  402355:	b9 18 00 00 00       	mov    $0x18,%ecx
  40235a:	e8 e1 0e 00 00       	call   403240 <__libirc_set_cpu_feature>
  40235f:	85 c0                	test   %eax,%eax
  402361:	0f 85 ab fc ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402367:	41 f7 c0 00 10 00 00 	test   $0x1000,%r8d
  40236e:	74 15                	je     402385 <__intel_cpu_features_init_body+0x7a5>
  402370:	48 89 e0             	mov    %rsp,%rax
  402373:	b9 13 00 00 00       	mov    $0x13,%ecx
  402378:	e8 c3 0e 00 00       	call   403240 <__libirc_set_cpu_feature>
  40237d:	85 c0                	test   %eax,%eax
  40237f:	0f 85 8d fc ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402385:	41 f6 c2 18          	test   $0x18,%r10b
  402389:	75 33                	jne    4023be <__intel_cpu_features_init_body+0x7de>
  40238b:	48 89 e0             	mov    %rsp,%rax
  40238e:	b9 01 00 00 00       	mov    $0x1,%ecx
  402393:	e8 a8 0e 00 00       	call   403240 <__libirc_set_cpu_feature>
  402398:	85 c0                	test   %eax,%eax
  40239a:	0f 85 72 fc ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4023a0:	41 f7 c1 00 40 00 00 	test   $0x4000,%r9d
  4023a7:	74 15                	je     4023be <__intel_cpu_features_init_body+0x7de>
  4023a9:	48 89 e0             	mov    %rsp,%rax
  4023ac:	b9 25 00 00 00       	mov    $0x25,%ecx
  4023b1:	e8 8a 0e 00 00       	call   403240 <__libirc_set_cpu_feature>
  4023b6:	85 c0                	test   %eax,%eax
  4023b8:	0f 85 54 fc ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4023be:	b8 07 00 00 00       	mov    $0x7,%eax
  4023c3:	b9 01 00 00 00       	mov    $0x1,%ecx
  4023c8:	0f a2                	cpuid
  4023ca:	89 c2                	mov    %eax,%edx
  4023cc:	f6 c2 10             	test   $0x10,%dl
  4023cf:	74 15                	je     4023e6 <__intel_cpu_features_init_body+0x806>
  4023d1:	48 89 e0             	mov    %rsp,%rax
  4023d4:	b9 41 00 00 00       	mov    $0x41,%ecx
  4023d9:	e8 62 0e 00 00       	call   403240 <__libirc_set_cpu_feature>
  4023de:	85 c0                	test   %eax,%eax
  4023e0:	0f 85 2c fc ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4023e6:	41 f6 c2 e0          	test   $0xe0,%r10b
  4023ea:	0f 85 f5 fb ff ff    	jne    401fe5 <__intel_cpu_features_init_body+0x405>
  4023f0:	48 89 e0             	mov    %rsp,%rax
  4023f3:	b9 01 00 00 00       	mov    $0x1,%ecx
  4023f8:	e8 43 0e 00 00       	call   403240 <__libirc_set_cpu_feature>
  4023fd:	85 c0                	test   %eax,%eax
  4023ff:	0f 85 0d fc ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402405:	41 f7 c1 00 00 01 00 	test   $0x10000,%r9d
  40240c:	74 15                	je     402423 <__intel_cpu_features_init_body+0x843>
  40240e:	48 89 e0             	mov    %rsp,%rax
  402411:	b9 19 00 00 00       	mov    $0x19,%ecx
  402416:	e8 25 0e 00 00       	call   403240 <__libirc_set_cpu_feature>
  40241b:	85 c0                	test   %eax,%eax
  40241d:	0f 85 ef fb ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402423:	41 f7 c1 00 00 00 10 	test   $0x10000000,%r9d
  40242a:	74 15                	je     402441 <__intel_cpu_features_init_body+0x861>
  40242c:	48 89 e0             	mov    %rsp,%rax
  40242f:	b9 23 00 00 00       	mov    $0x23,%ecx
  402434:	e8 07 0e 00 00       	call   403240 <__libirc_set_cpu_feature>
  402439:	85 c0                	test   %eax,%eax
  40243b:	0f 85 d1 fb ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402441:	41 f7 c1 00 00 00 08 	test   $0x8000000,%r9d
  402448:	74 15                	je     40245f <__intel_cpu_features_init_body+0x87f>
  40244a:	48 89 e0             	mov    %rsp,%rax
  40244d:	b9 21 00 00 00       	mov    $0x21,%ecx
  402452:	e8 e9 0d 00 00       	call   403240 <__libirc_set_cpu_feature>
  402457:	85 c0                	test   %eax,%eax
  402459:	0f 85 b3 fb ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  40245f:	41 f7 c1 00 00 00 04 	test   $0x4000000,%r9d
  402466:	74 15                	je     40247d <__intel_cpu_features_init_body+0x89d>
  402468:	48 89 e0             	mov    %rsp,%rax
  40246b:	b9 22 00 00 00       	mov    $0x22,%ecx
  402470:	e8 cb 0d 00 00       	call   403240 <__libirc_set_cpu_feature>
  402475:	85 c0                	test   %eax,%eax
  402477:	0f 85 95 fb ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  40247d:	41 f7 c1 00 00 02 00 	test   $0x20000,%r9d
  402484:	74 15                	je     40249b <__intel_cpu_features_init_body+0x8bb>
  402486:	48 89 e0             	mov    %rsp,%rax
  402489:	b9 1a 00 00 00       	mov    $0x1a,%ecx
  40248e:	e8 ad 0d 00 00       	call   403240 <__libirc_set_cpu_feature>
  402493:	85 c0                	test   %eax,%eax
  402495:	0f 85 77 fb ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  40249b:	41 f7 c1 00 00 00 40 	test   $0x40000000,%r9d
  4024a2:	74 15                	je     4024b9 <__intel_cpu_features_init_body+0x8d9>
  4024a4:	48 89 e0             	mov    %rsp,%rax
  4024a7:	b9 26 00 00 00       	mov    $0x26,%ecx
  4024ac:	e8 8f 0d 00 00       	call   403240 <__libirc_set_cpu_feature>
  4024b1:	85 c0                	test   %eax,%eax
  4024b3:	0f 85 59 fb ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4024b9:	45 85 c9             	test   %r9d,%r9d
  4024bc:	0f 88 b5 01 00 00    	js     402677 <__intel_cpu_features_init_body+0xa97>
  4024c2:	41 f7 c1 00 00 20 00 	test   $0x200000,%r9d
  4024c9:	74 15                	je     4024e0 <__intel_cpu_features_init_body+0x900>
  4024cb:	48 89 e0             	mov    %rsp,%rax
  4024ce:	b9 1f 00 00 00       	mov    $0x1f,%ecx
  4024d3:	e8 68 0d 00 00       	call   403240 <__libirc_set_cpu_feature>
  4024d8:	85 c0                	test   %eax,%eax
  4024da:	0f 85 32 fb ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4024e0:	40 f6 c7 02          	test   $0x2,%dil
  4024e4:	74 15                	je     4024fb <__intel_cpu_features_init_body+0x91b>
  4024e6:	48 89 e0             	mov    %rsp,%rax
  4024e9:	b9 28 00 00 00       	mov    $0x28,%ecx
  4024ee:	e8 4d 0d 00 00       	call   403240 <__libirc_set_cpu_feature>
  4024f3:	85 c0                	test   %eax,%eax
  4024f5:	0f 85 17 fb ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4024fb:	f7 c7 00 40 00 00    	test   $0x4000,%edi
  402501:	74 15                	je     402518 <__intel_cpu_features_init_body+0x938>
  402503:	48 89 e0             	mov    %rsp,%rax
  402506:	b9 2b 00 00 00       	mov    $0x2b,%ecx
  40250b:	e8 30 0d 00 00       	call   403240 <__libirc_set_cpu_feature>
  402510:	85 c0                	test   %eax,%eax
  402512:	0f 85 fa fa ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402518:	40 f6 c6 04          	test   $0x4,%sil
  40251c:	74 15                	je     402533 <__intel_cpu_features_init_body+0x953>
  40251e:	48 89 e0             	mov    %rsp,%rax
  402521:	b9 2a 00 00 00       	mov    $0x2a,%ecx
  402526:	e8 15 0d 00 00       	call   403240 <__libirc_set_cpu_feature>
  40252b:	85 c0                	test   %eax,%eax
  40252d:	0f 85 df fa ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402533:	40 f6 c6 08          	test   $0x8,%sil
  402537:	74 15                	je     40254e <__intel_cpu_features_init_body+0x96e>
  402539:	48 89 e0             	mov    %rsp,%rax
  40253c:	b9 29 00 00 00       	mov    $0x29,%ecx
  402541:	e8 fa 0c 00 00       	call   403240 <__libirc_set_cpu_feature>
  402546:	85 c0                	test   %eax,%eax
  402548:	0f 85 c4 fa ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  40254e:	f7 c7 00 10 00 00    	test   $0x1000,%edi
  402554:	74 15                	je     40256b <__intel_cpu_features_init_body+0x98b>
  402556:	48 89 e0             	mov    %rsp,%rax
  402559:	b9 2c 00 00 00       	mov    $0x2c,%ecx
  40255e:	e8 dd 0c 00 00       	call   403240 <__libirc_set_cpu_feature>
  402563:	85 c0                	test   %eax,%eax
  402565:	0f 85 a7 fa ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  40256b:	40 f6 c7 40          	test   $0x40,%dil
  40256f:	74 15                	je     402586 <__intel_cpu_features_init_body+0x9a6>
  402571:	48 89 e0             	mov    %rsp,%rax
  402574:	b9 2d 00 00 00       	mov    $0x2d,%ecx
  402579:	e8 c2 0c 00 00       	call   403240 <__libirc_set_cpu_feature>
  40257e:	85 c0                	test   %eax,%eax
  402580:	0f 85 8c fa ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402586:	f7 c7 00 08 00 00    	test   $0x800,%edi
  40258c:	74 15                	je     4025a3 <__intel_cpu_features_init_body+0x9c3>
  40258e:	48 89 e0             	mov    %rsp,%rax
  402591:	b9 31 00 00 00       	mov    $0x31,%ecx
  402596:	e8 a5 0c 00 00       	call   403240 <__libirc_set_cpu_feature>
  40259b:	85 c0                	test   %eax,%eax
  40259d:	0f 85 6f fa ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4025a3:	f6 c2 20             	test   $0x20,%dl
  4025a6:	74 15                	je     4025bd <__intel_cpu_features_init_body+0x9dd>
  4025a8:	48 89 e0             	mov    %rsp,%rax
  4025ab:	b9 3f 00 00 00       	mov    $0x3f,%ecx
  4025b0:	e8 8b 0c 00 00       	call   403240 <__libirc_set_cpu_feature>
  4025b5:	85 c0                	test   %eax,%eax
  4025b7:	0f 85 55 fa ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4025bd:	f7 c6 00 00 80 00    	test   $0x800000,%esi
  4025c3:	74 15                	je     4025da <__intel_cpu_features_init_body+0x9fa>
  4025c5:	48 89 e0             	mov    %rsp,%rax
  4025c8:	b9 3a 00 00 00       	mov    $0x3a,%ecx
  4025cd:	e8 6e 0c 00 00       	call   403240 <__libirc_set_cpu_feature>
  4025d2:	85 c0                	test   %eax,%eax
  4025d4:	0f 85 38 fa ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4025da:	f7 c6 00 01 00 00    	test   $0x100,%esi
  4025e0:	0f 84 ff f9 ff ff    	je     401fe5 <__intel_cpu_features_init_body+0x405>
  4025e6:	48 89 e0             	mov    %rsp,%rax
  4025e9:	b9 39 00 00 00       	mov    $0x39,%ecx
  4025ee:	e8 4d 0c 00 00       	call   403240 <__libirc_set_cpu_feature>
  4025f3:	85 c0                	test   %eax,%eax
  4025f5:	0f 85 17 fa ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  4025fb:	e9 e5 f9 ff ff       	jmp    401fe5 <__intel_cpu_features_init_body+0x405>
  402600:	0f 28 04 24          	movaps (%rsp),%xmm0
  402604:	83 fd 01             	cmp    $0x1,%ebp
  402607:	75 07                	jne    402610 <__intel_cpu_features_init_body+0xa30>
  402609:	0f 29 05 c0 4a 00 00 	movaps %xmm0,0x4ac0(%rip)        # 4070d0 <__intel_cpu_feature_indicator>
  402610:	48 c7 c0 e0 70 40 00 	mov    $0x4070e0,%rax
  402617:	0f 29 00             	movaps %xmm0,(%rax)
  40261a:	e9 f1 f9 ff ff       	jmp    402010 <__intel_cpu_features_init_body+0x430>
  40261f:	48 89 e0             	mov    %rsp,%rax
  402622:	b9 10 00 00 00       	mov    $0x10,%ecx
  402627:	e8 14 0c 00 00       	call   403240 <__libirc_set_cpu_feature>
  40262c:	85 c0                	test   %eax,%eax
  40262e:	0f 85 de f9 ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402634:	f7 c7 00 02 00 00    	test   $0x200,%edi
  40263a:	74 15                	je     402651 <__intel_cpu_features_init_body+0xa71>
  40263c:	48 89 e0             	mov    %rsp,%rax
  40263f:	b9 2f 00 00 00       	mov    $0x2f,%ecx
  402644:	e8 f7 0b 00 00       	call   403240 <__libirc_set_cpu_feature>
  402649:	85 c0                	test   %eax,%eax
  40264b:	0f 85 c1 f9 ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402651:	f7 c7 00 04 00 00    	test   $0x400,%edi
  402657:	0f 84 d1 fc ff ff    	je     40232e <__intel_cpu_features_init_body+0x74e>
  40265d:	48 89 e0             	mov    %rsp,%rax
  402660:	b9 30 00 00 00       	mov    $0x30,%ecx
  402665:	e8 d6 0b 00 00       	call   403240 <__libirc_set_cpu_feature>
  40266a:	85 c0                	test   %eax,%eax
  40266c:	0f 85 a0 f9 ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  402672:	e9 b7 fc ff ff       	jmp    40232e <__intel_cpu_features_init_body+0x74e>
  402677:	48 89 e0             	mov    %rsp,%rax
  40267a:	b9 27 00 00 00       	mov    $0x27,%ecx
  40267f:	e8 bc 0b 00 00       	call   403240 <__libirc_set_cpu_feature>
  402684:	85 c0                	test   %eax,%eax
  402686:	0f 85 86 f9 ff ff    	jne    402012 <__intel_cpu_features_init_body+0x432>
  40268c:	e9 31 fe ff ff       	jmp    4024c2 <__intel_cpu_features_init_body+0x8e2>
  402691:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  402698:	00 00 00 
  40269b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004026a0 <__intel_cpu_features_init_x>:
  4026a0:	f3 0f 1e fa          	endbr64
  4026a4:	50                   	push   %rax
  4026a5:	31 c0                	xor    %eax,%eax
  4026a7:	e8 34 f5 ff ff       	call   401be0 <__intel_cpu_features_init_body>
  4026ac:	48 83 c4 08          	add    $0x8,%rsp
  4026b0:	c3                   	ret
  4026b1:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4026b8:	00 00 00 
  4026bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004026c0 <__libirc_get_feature_name>:
  4026c0:	f3 0f 1e fa          	endbr64
  4026c4:	50                   	push   %rax
  4026c5:	80 3d 24 4a 00 00 00 	cmpb   $0x0,0x4a24(%rip)        # 4070f0 <__libirc_isa_info_initialized>
  4026cc:	75 05                	jne    4026d3 <__libirc_get_feature_name+0x13>
  4026ce:	e8 1d 00 00 00       	call   4026f0 <__libirc_isa_init_once>
  4026d3:	89 f8                	mov    %edi,%eax
  4026d5:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  4026d9:	48 8d 0d 20 4a 00 00 	lea    0x4a20(%rip),%rcx        # 407100 <proc_info_features>
  4026e0:	48 8b 04 c1          	mov    (%rcx,%rax,8),%rax
  4026e4:	59                   	pop    %rcx
  4026e5:	c3                   	ret
  4026e6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4026ed:	00 00 00 

00000000004026f0 <__libirc_isa_init_once>:
  4026f0:	51                   	push   %rcx
  4026f1:	80 3d f8 49 00 00 00 	cmpb   $0x0,0x49f8(%rip)        # 4070f0 <__libirc_isa_info_initialized>
  4026f8:	0f 85 aa 0a 00 00    	jne    4031a8 <__libirc_isa_init_once+0xab8>
  4026fe:	b8 c8 00 00 00       	mov    $0xc8,%eax
  402703:	48 8d 0d f6 49 00 00 	lea    0x49f6(%rip),%rcx        # 407100 <proc_info_features>
  40270a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  402710:	c7 84 08 58 ff ff ff 	movl   $0xffffffff,-0xa8(%rax,%rcx,1)
  402717:	ff ff ff ff 
  40271b:	c7 84 08 70 ff ff ff 	movl   $0xffffffff,-0x90(%rax,%rcx,1)
  402722:	ff ff ff ff 
  402726:	c7 44 08 88 ff ff ff 	movl   $0xffffffff,-0x78(%rax,%rcx,1)
  40272d:	ff 
  40272e:	c7 44 08 a0 ff ff ff 	movl   $0xffffffff,-0x60(%rax,%rcx,1)
  402735:	ff 
  402736:	c7 44 08 b8 ff ff ff 	movl   $0xffffffff,-0x48(%rax,%rcx,1)
  40273d:	ff 
  40273e:	c7 44 08 d0 ff ff ff 	movl   $0xffffffff,-0x30(%rax,%rcx,1)
  402745:	ff 
  402746:	c7 44 08 e8 ff ff ff 	movl   $0xffffffff,-0x18(%rax,%rcx,1)
  40274d:	ff 
  40274e:	c7 04 08 ff ff ff ff 	movl   $0xffffffff,(%rax,%rcx,1)
  402755:	48 05 c0 00 00 00    	add    $0xc0,%rax
  40275b:	48 3d c8 06 00 00    	cmp    $0x6c8,%rax
  402761:	75 ad                	jne    402710 <__libirc_isa_init_once+0x20>
  402763:	c7 05 b3 4f 00 00 ff 	movl   $0xffffffff,0x4fb3(%rip)        # 407720 <proc_info_features+0x620>
  40276a:	ff ff ff 
  40276d:	c7 05 c1 4f 00 00 ff 	movl   $0xffffffff,0x4fc1(%rip)        # 407738 <proc_info_features+0x638>
  402774:	ff ff ff 
  402777:	c7 05 cf 4f 00 00 ff 	movl   $0xffffffff,0x4fcf(%rip)        # 407750 <proc_info_features+0x650>
  40277e:	ff ff ff 
  402781:	c7 05 dd 4f 00 00 ff 	movl   $0xffffffff,0x4fdd(%rip)        # 407768 <proc_info_features+0x668>
  402788:	ff ff ff 
  40278b:	c7 05 eb 4f 00 00 ff 	movl   $0xffffffff,0x4feb(%rip)        # 407780 <proc_info_features+0x680>
  402792:	ff ff ff 
  402795:	c7 05 f9 4f 00 00 ff 	movl   $0xffffffff,0x4ff9(%rip)        # 407798 <proc_info_features+0x698>
  40279c:	ff ff ff 
  40279f:	48 8d 05 03 19 00 00 	lea    0x1903(%rip),%rax        # 4040a9 <_IO_stdin_used+0xa9>
  4027a6:	48 89 05 6b 49 00 00 	mov    %rax,0x496b(%rip)        # 407118 <proc_info_features+0x18>
  4027ad:	c7 05 69 49 00 00 00 	movl   $0x0,0x4969(%rip)        # 407120 <proc_info_features+0x20>
  4027b4:	00 00 00 
  4027b7:	48 8d 05 f8 18 00 00 	lea    0x18f8(%rip),%rax        # 4040b6 <_IO_stdin_used+0xb6>
  4027be:	48 89 05 6b 49 00 00 	mov    %rax,0x496b(%rip)        # 407130 <proc_info_features+0x30>
  4027c5:	c7 05 69 49 00 00 01 	movl   $0x1,0x4969(%rip)        # 407138 <proc_info_features+0x38>
  4027cc:	00 00 00 
  4027cf:	48 8d 05 e4 18 00 00 	lea    0x18e4(%rip),%rax        # 4040ba <_IO_stdin_used+0xba>
  4027d6:	48 89 05 6b 49 00 00 	mov    %rax,0x496b(%rip)        # 407148 <proc_info_features+0x48>
  4027dd:	c7 05 69 49 00 00 02 	movl   $0x2,0x4969(%rip)        # 407150 <proc_info_features+0x50>
  4027e4:	00 00 00 
  4027e7:	c7 05 77 49 00 00 03 	movl   $0x3,0x4977(%rip)        # 407168 <proc_info_features+0x68>
  4027ee:	00 00 00 
  4027f1:	48 8d 05 c7 18 00 00 	lea    0x18c7(%rip),%rax        # 4040bf <_IO_stdin_used+0xbf>
  4027f8:	48 89 05 71 49 00 00 	mov    %rax,0x4971(%rip)        # 407170 <proc_info_features+0x70>
  4027ff:	48 8d 05 bd 18 00 00 	lea    0x18bd(%rip),%rax        # 4040c3 <_IO_stdin_used+0xc3>
  402806:	48 89 05 53 49 00 00 	mov    %rax,0x4953(%rip)        # 407160 <proc_info_features+0x60>
  40280d:	48 8d 05 b3 18 00 00 	lea    0x18b3(%rip),%rax        # 4040c7 <_IO_stdin_used+0xc7>
  402814:	48 89 05 5d 49 00 00 	mov    %rax,0x495d(%rip)        # 407178 <proc_info_features+0x78>
  40281b:	c7 05 5b 49 00 00 04 	movl   $0x4,0x495b(%rip)        # 407180 <proc_info_features+0x80>
  402822:	00 00 00 
  402825:	c7 05 69 49 00 00 05 	movl   $0x5,0x4969(%rip)        # 407198 <proc_info_features+0x98>
  40282c:	00 00 00 
  40282f:	48 8d 05 98 18 00 00 	lea    0x1898(%rip),%rax        # 4040ce <_IO_stdin_used+0xce>
  402836:	48 89 05 63 49 00 00 	mov    %rax,0x4963(%rip)        # 4071a0 <proc_info_features+0xa0>
  40283d:	48 8d 05 8e 18 00 00 	lea    0x188e(%rip),%rax        # 4040d2 <_IO_stdin_used+0xd2>
  402844:	48 89 05 45 49 00 00 	mov    %rax,0x4945(%rip)        # 407190 <proc_info_features+0x90>
  40284b:	c7 05 5b 49 00 00 06 	movl   $0x6,0x495b(%rip)        # 4071b0 <proc_info_features+0xb0>
  402852:	00 00 00 
  402855:	48 8d 05 7a 18 00 00 	lea    0x187a(%rip),%rax        # 4040d6 <_IO_stdin_used+0xd6>
  40285c:	48 89 05 55 49 00 00 	mov    %rax,0x4955(%rip)        # 4071b8 <proc_info_features+0xb8>
  402863:	48 8d 05 71 18 00 00 	lea    0x1871(%rip),%rax        # 4040db <_IO_stdin_used+0xdb>
  40286a:	48 89 05 37 49 00 00 	mov    %rax,0x4937(%rip)        # 4071a8 <proc_info_features+0xa8>
  402871:	c7 05 4d 49 00 00 07 	movl   $0x7,0x494d(%rip)        # 4071c8 <proc_info_features+0xc8>
  402878:	00 00 00 
  40287b:	48 8d 05 5f 18 00 00 	lea    0x185f(%rip),%rax        # 4040e1 <_IO_stdin_used+0xe1>
  402882:	48 89 05 47 49 00 00 	mov    %rax,0x4947(%rip)        # 4071d0 <proc_info_features+0xd0>
  402889:	48 8d 05 57 18 00 00 	lea    0x1857(%rip),%rax        # 4040e7 <_IO_stdin_used+0xe7>
  402890:	48 89 05 29 49 00 00 	mov    %rax,0x4929(%rip)        # 4071c0 <proc_info_features+0xc0>
  402897:	c7 05 3f 49 00 00 08 	movl   $0x8,0x493f(%rip)        # 4071e0 <proc_info_features+0xe0>
  40289e:	00 00 00 
  4028a1:	48 8d 05 38 18 00 00 	lea    0x1838(%rip),%rax        # 4040e0 <_IO_stdin_used+0xe0>
  4028a8:	48 89 05 39 49 00 00 	mov    %rax,0x4939(%rip)        # 4071e8 <proc_info_features+0xe8>
  4028af:	48 8d 05 30 18 00 00 	lea    0x1830(%rip),%rax        # 4040e6 <_IO_stdin_used+0xe6>
  4028b6:	48 89 05 1b 49 00 00 	mov    %rax,0x491b(%rip)        # 4071d8 <proc_info_features+0xd8>
  4028bd:	c7 05 31 49 00 00 09 	movl   $0x9,0x4931(%rip)        # 4071f8 <proc_info_features+0xf8>
  4028c4:	00 00 00 
  4028c7:	48 8d 05 1e 18 00 00 	lea    0x181e(%rip),%rax        # 4040ec <_IO_stdin_used+0xec>
  4028ce:	48 89 05 2b 49 00 00 	mov    %rax,0x492b(%rip)        # 407200 <proc_info_features+0x100>
  4028d5:	48 8d 05 17 18 00 00 	lea    0x1817(%rip),%rax        # 4040f3 <_IO_stdin_used+0xf3>
  4028dc:	48 89 05 0d 49 00 00 	mov    %rax,0x490d(%rip)        # 4071f0 <proc_info_features+0xf0>
  4028e3:	c7 05 23 49 00 00 0a 	movl   $0xa,0x4923(%rip)        # 407210 <proc_info_features+0x110>
  4028ea:	00 00 00 
  4028ed:	48 8d 05 06 18 00 00 	lea    0x1806(%rip),%rax        # 4040fa <_IO_stdin_used+0xfa>
  4028f4:	48 89 05 1d 49 00 00 	mov    %rax,0x491d(%rip)        # 407218 <proc_info_features+0x118>
  4028fb:	48 8d 05 fd 17 00 00 	lea    0x17fd(%rip),%rax        # 4040ff <_IO_stdin_used+0xff>
  402902:	48 89 05 ff 48 00 00 	mov    %rax,0x48ff(%rip)        # 407208 <proc_info_features+0x108>
  402909:	c7 05 15 49 00 00 0b 	movl   $0xb,0x4915(%rip)        # 407228 <proc_info_features+0x128>
  402910:	00 00 00 
  402913:	48 8d 05 ec 17 00 00 	lea    0x17ec(%rip),%rax        # 404106 <_IO_stdin_used+0x106>
  40291a:	48 89 05 0f 49 00 00 	mov    %rax,0x490f(%rip)        # 407230 <proc_info_features+0x130>
  402921:	48 8d 05 e4 17 00 00 	lea    0x17e4(%rip),%rax        # 40410c <_IO_stdin_used+0x10c>
  402928:	48 89 05 f1 48 00 00 	mov    %rax,0x48f1(%rip)        # 407220 <proc_info_features+0x120>
  40292f:	c7 05 07 49 00 00 0c 	movl   $0xc,0x4907(%rip)        # 407240 <proc_info_features+0x140>
  402936:	00 00 00 
  402939:	48 8d 05 d2 17 00 00 	lea    0x17d2(%rip),%rax        # 404112 <_IO_stdin_used+0x112>
  402940:	48 89 05 01 49 00 00 	mov    %rax,0x4901(%rip)        # 407248 <proc_info_features+0x148>
  402947:	48 8d 05 cb 17 00 00 	lea    0x17cb(%rip),%rax        # 404119 <_IO_stdin_used+0x119>
  40294e:	48 89 05 e3 48 00 00 	mov    %rax,0x48e3(%rip)        # 407238 <proc_info_features+0x138>
  402955:	c7 05 f9 48 00 00 0d 	movl   $0xd,0x48f9(%rip)        # 407258 <proc_info_features+0x158>
  40295c:	00 00 00 
  40295f:	48 8d 05 ba 17 00 00 	lea    0x17ba(%rip),%rax        # 404120 <_IO_stdin_used+0x120>
  402966:	48 89 05 f3 48 00 00 	mov    %rax,0x48f3(%rip)        # 407260 <proc_info_features+0x160>
  40296d:	48 8d 05 8c 19 00 00 	lea    0x198c(%rip),%rax        # 404300 <_IO_stdin_used+0x300>
  402974:	48 89 05 d5 48 00 00 	mov    %rax,0x48d5(%rip)        # 407250 <proc_info_features+0x150>
  40297b:	c7 05 eb 48 00 00 0e 	movl   $0xe,0x48eb(%rip)        # 407270 <proc_info_features+0x170>
  402982:	00 00 00 
  402985:	48 8d 05 5f 19 00 00 	lea    0x195f(%rip),%rax        # 4042eb <_IO_stdin_used+0x2eb>
  40298c:	48 89 05 e5 48 00 00 	mov    %rax,0x48e5(%rip)        # 407278 <proc_info_features+0x178>
  402993:	48 8d 05 56 19 00 00 	lea    0x1956(%rip),%rax        # 4042f0 <_IO_stdin_used+0x2f0>
  40299a:	48 89 05 c7 48 00 00 	mov    %rax,0x48c7(%rip)        # 407268 <proc_info_features+0x168>
  4029a1:	c7 05 dd 48 00 00 10 	movl   $0x10,0x48dd(%rip)        # 407288 <proc_info_features+0x188>
  4029a8:	00 00 00 
  4029ab:	48 8d 05 75 17 00 00 	lea    0x1775(%rip),%rax        # 404127 <_IO_stdin_used+0x127>
  4029b2:	48 89 05 d7 48 00 00 	mov    %rax,0x48d7(%rip)        # 407290 <proc_info_features+0x190>
  4029b9:	48 8d 05 6b 17 00 00 	lea    0x176b(%rip),%rax        # 40412b <_IO_stdin_used+0x12b>
  4029c0:	48 89 05 b9 48 00 00 	mov    %rax,0x48b9(%rip)        # 407280 <proc_info_features+0x180>
  4029c7:	c7 05 cf 48 00 00 0f 	movl   $0xf,0x48cf(%rip)        # 4072a0 <proc_info_features+0x1a0>
  4029ce:	00 00 00 
  4029d1:	48 8d 05 57 17 00 00 	lea    0x1757(%rip),%rax        # 40412f <_IO_stdin_used+0x12f>
  4029d8:	48 89 05 c9 48 00 00 	mov    %rax,0x48c9(%rip)        # 4072a8 <proc_info_features+0x1a8>
  4029df:	48 8d 05 4e 17 00 00 	lea    0x174e(%rip),%rax        # 404134 <_IO_stdin_used+0x134>
  4029e6:	48 89 05 ab 48 00 00 	mov    %rax,0x48ab(%rip)        # 407298 <proc_info_features+0x198>
  4029ed:	c7 05 c1 48 00 00 11 	movl   $0x11,0x48c1(%rip)        # 4072b8 <proc_info_features+0x1b8>
  4029f4:	00 00 00 
  4029f7:	48 8d 05 3b 17 00 00 	lea    0x173b(%rip),%rax        # 404139 <_IO_stdin_used+0x139>
  4029fe:	48 89 05 bb 48 00 00 	mov    %rax,0x48bb(%rip)        # 4072c0 <proc_info_features+0x1c0>
  402a05:	48 8d 05 33 17 00 00 	lea    0x1733(%rip),%rax        # 40413f <_IO_stdin_used+0x13f>
  402a0c:	48 89 05 9d 48 00 00 	mov    %rax,0x489d(%rip)        # 4072b0 <proc_info_features+0x1b0>
  402a13:	c7 05 b3 48 00 00 12 	movl   $0x12,0x48b3(%rip)        # 4072d0 <proc_info_features+0x1d0>
  402a1a:	00 00 00 
  402a1d:	48 8d 05 a0 17 00 00 	lea    0x17a0(%rip),%rax        # 4041c4 <_IO_stdin_used+0x1c4>
  402a24:	48 89 05 ad 48 00 00 	mov    %rax,0x48ad(%rip)        # 4072d8 <proc_info_features+0x1d8>
  402a2b:	48 8d 05 13 17 00 00 	lea    0x1713(%rip),%rax        # 404145 <_IO_stdin_used+0x145>
  402a32:	48 89 05 8f 48 00 00 	mov    %rax,0x488f(%rip)        # 4072c8 <proc_info_features+0x1c8>
  402a39:	c7 05 a5 48 00 00 13 	movl   $0x13,0x48a5(%rip)        # 4072e8 <proc_info_features+0x1e8>
  402a40:	00 00 00 
  402a43:	48 8d 05 fc 17 00 00 	lea    0x17fc(%rip),%rax        # 404246 <_IO_stdin_used+0x246>
  402a4a:	48 89 05 9f 48 00 00 	mov    %rax,0x489f(%rip)        # 4072f0 <proc_info_features+0x1f0>
  402a51:	48 8d 05 f9 17 00 00 	lea    0x17f9(%rip),%rax        # 404251 <_IO_stdin_used+0x251>
  402a58:	48 89 05 81 48 00 00 	mov    %rax,0x4881(%rip)        # 4072e0 <proc_info_features+0x1e0>
  402a5f:	c7 05 97 48 00 00 14 	movl   $0x14,0x4897(%rip)        # 407300 <proc_info_features+0x200>
  402a66:	00 00 00 
  402a69:	48 8d 05 d9 16 00 00 	lea    0x16d9(%rip),%rax        # 404149 <_IO_stdin_used+0x149>
  402a70:	48 89 05 91 48 00 00 	mov    %rax,0x4891(%rip)        # 407308 <proc_info_features+0x208>
  402a77:	48 8d 05 d1 16 00 00 	lea    0x16d1(%rip),%rax        # 40414f <_IO_stdin_used+0x14f>
  402a7e:	48 89 05 73 48 00 00 	mov    %rax,0x4873(%rip)        # 4072f8 <proc_info_features+0x1f8>
  402a85:	c7 05 89 48 00 00 15 	movl   $0x15,0x4889(%rip)        # 407318 <proc_info_features+0x218>
  402a8c:	00 00 00 
  402a8f:	48 8d 05 bf 16 00 00 	lea    0x16bf(%rip),%rax        # 404155 <_IO_stdin_used+0x155>
  402a96:	48 89 05 83 48 00 00 	mov    %rax,0x4883(%rip)        # 407320 <proc_info_features+0x220>
  402a9d:	48 8d 05 b5 16 00 00 	lea    0x16b5(%rip),%rax        # 404159 <_IO_stdin_used+0x159>
  402aa4:	48 89 05 65 48 00 00 	mov    %rax,0x4865(%rip)        # 407310 <proc_info_features+0x210>
  402aab:	c7 05 7b 48 00 00 16 	movl   $0x16,0x487b(%rip)        # 407330 <proc_info_features+0x230>
  402ab2:	00 00 00 
  402ab5:	48 8d 05 a1 16 00 00 	lea    0x16a1(%rip),%rax        # 40415d <_IO_stdin_used+0x15d>
  402abc:	48 89 05 75 48 00 00 	mov    %rax,0x4875(%rip)        # 407338 <proc_info_features+0x238>
  402ac3:	48 8d 05 97 16 00 00 	lea    0x1697(%rip),%rax        # 404161 <_IO_stdin_used+0x161>
  402aca:	48 89 05 57 48 00 00 	mov    %rax,0x4857(%rip)        # 407328 <proc_info_features+0x228>
  402ad1:	c7 05 6d 48 00 00 17 	movl   $0x17,0x486d(%rip)        # 407348 <proc_info_features+0x248>
  402ad8:	00 00 00 
  402adb:	48 8d 05 83 16 00 00 	lea    0x1683(%rip),%rax        # 404165 <_IO_stdin_used+0x165>
  402ae2:	48 89 05 67 48 00 00 	mov    %rax,0x4867(%rip)        # 407350 <proc_info_features+0x250>
  402ae9:	48 8d 05 7a 16 00 00 	lea    0x167a(%rip),%rax        # 40416a <_IO_stdin_used+0x16a>
  402af0:	48 89 05 49 48 00 00 	mov    %rax,0x4849(%rip)        # 407340 <proc_info_features+0x240>
  402af7:	c7 05 5f 48 00 00 1b 	movl   $0x1b,0x485f(%rip)        # 407360 <proc_info_features+0x260>
  402afe:	00 00 00 
  402b01:	48 8d 05 67 16 00 00 	lea    0x1667(%rip),%rax        # 40416f <_IO_stdin_used+0x16f>
  402b08:	48 89 05 59 48 00 00 	mov    %rax,0x4859(%rip)        # 407368 <proc_info_features+0x268>
  402b0f:	48 8d 05 61 16 00 00 	lea    0x1661(%rip),%rax        # 404177 <_IO_stdin_used+0x177>
  402b16:	48 89 05 3b 48 00 00 	mov    %rax,0x483b(%rip)        # 407358 <proc_info_features+0x258>
  402b1d:	c7 05 51 48 00 00 18 	movl   $0x18,0x4851(%rip)        # 407378 <proc_info_features+0x278>
  402b24:	00 00 00 
  402b27:	48 8d 05 51 16 00 00 	lea    0x1651(%rip),%rax        # 40417f <_IO_stdin_used+0x17f>
  402b2e:	48 89 05 4b 48 00 00 	mov    %rax,0x484b(%rip)        # 407380 <proc_info_features+0x280>
  402b35:	48 8d 05 4c 16 00 00 	lea    0x164c(%rip),%rax        # 404188 <_IO_stdin_used+0x188>
  402b3c:	48 89 05 2d 48 00 00 	mov    %rax,0x482d(%rip)        # 407370 <proc_info_features+0x270>
  402b43:	c7 05 43 48 00 00 19 	movl   $0x19,0x4843(%rip)        # 407390 <proc_info_features+0x290>
  402b4a:	00 00 00 
  402b4d:	48 8d 05 3d 16 00 00 	lea    0x163d(%rip),%rax        # 404191 <_IO_stdin_used+0x191>
  402b54:	48 89 05 3d 48 00 00 	mov    %rax,0x483d(%rip)        # 407398 <proc_info_features+0x298>
  402b5b:	48 8d 05 37 16 00 00 	lea    0x1637(%rip),%rax        # 404199 <_IO_stdin_used+0x199>
  402b62:	48 89 05 1f 48 00 00 	mov    %rax,0x481f(%rip)        # 407388 <proc_info_features+0x288>
  402b69:	48 8d 05 31 16 00 00 	lea    0x1631(%rip),%rax        # 4041a1 <_IO_stdin_used+0x1a1>
  402b70:	48 89 05 29 48 00 00 	mov    %rax,0x4829(%rip)        # 4073a0 <proc_info_features+0x2a0>
  402b77:	c7 05 27 48 00 00 1a 	movl   $0x1a,0x4827(%rip)        # 4073a8 <proc_info_features+0x2a8>
  402b7e:	00 00 00 
  402b81:	c7 05 35 48 00 00 1c 	movl   $0x1c,0x4835(%rip)        # 4073c0 <proc_info_features+0x2c0>
  402b88:	00 00 00 
  402b8b:	48 8d 05 15 16 00 00 	lea    0x1615(%rip),%rax        # 4041a7 <_IO_stdin_used+0x1a7>
  402b92:	48 89 05 2f 48 00 00 	mov    %rax,0x482f(%rip)        # 4073c8 <proc_info_features+0x2c8>
  402b99:	48 8d 05 0b 16 00 00 	lea    0x160b(%rip),%rax        # 4041ab <_IO_stdin_used+0x1ab>
  402ba0:	48 89 05 11 48 00 00 	mov    %rax,0x4811(%rip)        # 4073b8 <proc_info_features+0x2b8>
  402ba7:	c7 05 27 48 00 00 1d 	movl   $0x1d,0x4827(%rip)        # 4073d8 <proc_info_features+0x2d8>
  402bae:	00 00 00 
  402bb1:	48 8d 05 f7 15 00 00 	lea    0x15f7(%rip),%rax        # 4041af <_IO_stdin_used+0x1af>
  402bb8:	48 89 05 21 48 00 00 	mov    %rax,0x4821(%rip)        # 4073e0 <proc_info_features+0x2e0>
  402bbf:	48 8d 05 f0 15 00 00 	lea    0x15f0(%rip),%rax        # 4041b6 <_IO_stdin_used+0x1b6>
  402bc6:	48 89 05 03 48 00 00 	mov    %rax,0x4803(%rip)        # 4073d0 <proc_info_features+0x2d0>
  402bcd:	c7 05 19 48 00 00 1e 	movl   $0x1e,0x4819(%rip)        # 4073f0 <proc_info_features+0x2f0>
  402bd4:	00 00 00 
  402bd7:	48 8d 05 df 15 00 00 	lea    0x15df(%rip),%rax        # 4041bd <_IO_stdin_used+0x1bd>
  402bde:	48 89 05 13 48 00 00 	mov    %rax,0x4813(%rip)        # 4073f8 <proc_info_features+0x2f8>
  402be5:	48 8d 05 dc 15 00 00 	lea    0x15dc(%rip),%rax        # 4041c8 <_IO_stdin_used+0x1c8>
  402bec:	48 89 05 f5 47 00 00 	mov    %rax,0x47f5(%rip)        # 4073e8 <proc_info_features+0x2e8>
  402bf3:	c7 05 0b 48 00 00 ff 	movl   $0xffffffff,0x480b(%rip)        # 407408 <proc_info_features+0x308>
  402bfa:	ff ff ff 
  402bfd:	c7 05 19 48 00 00 20 	movl   $0x20,0x4819(%rip)        # 407420 <proc_info_features+0x320>
  402c04:	00 00 00 
  402c07:	48 8d 05 c7 15 00 00 	lea    0x15c7(%rip),%rax        # 4041d5 <_IO_stdin_used+0x1d5>
  402c0e:	48 89 05 13 48 00 00 	mov    %rax,0x4813(%rip)        # 407428 <proc_info_features+0x328>
  402c15:	48 8d 05 c2 15 00 00 	lea    0x15c2(%rip),%rax        # 4041de <_IO_stdin_used+0x1de>
  402c1c:	48 89 05 f5 47 00 00 	mov    %rax,0x47f5(%rip)        # 407418 <proc_info_features+0x318>
  402c23:	c7 05 0b 48 00 00 21 	movl   $0x21,0x480b(%rip)        # 407438 <proc_info_features+0x338>
  402c2a:	00 00 00 
  402c2d:	48 8d 05 b3 15 00 00 	lea    0x15b3(%rip),%rax        # 4041e7 <_IO_stdin_used+0x1e7>
  402c34:	48 89 05 05 48 00 00 	mov    %rax,0x4805(%rip)        # 407440 <proc_info_features+0x340>
  402c3b:	48 8d 05 ae 15 00 00 	lea    0x15ae(%rip),%rax        # 4041f0 <_IO_stdin_used+0x1f0>
  402c42:	48 89 05 e7 47 00 00 	mov    %rax,0x47e7(%rip)        # 407430 <proc_info_features+0x330>
  402c49:	c7 05 fd 47 00 00 22 	movl   $0x22,0x47fd(%rip)        # 407450 <proc_info_features+0x350>
  402c50:	00 00 00 
  402c53:	48 8d 05 9f 15 00 00 	lea    0x159f(%rip),%rax        # 4041f9 <_IO_stdin_used+0x1f9>
  402c5a:	48 89 05 f7 47 00 00 	mov    %rax,0x47f7(%rip)        # 407458 <proc_info_features+0x358>
  402c61:	48 8d 05 9a 15 00 00 	lea    0x159a(%rip),%rax        # 404202 <_IO_stdin_used+0x202>
  402c68:	48 89 05 d9 47 00 00 	mov    %rax,0x47d9(%rip)        # 407448 <proc_info_features+0x348>
  402c6f:	c7 05 ef 47 00 00 23 	movl   $0x23,0x47ef(%rip)        # 407468 <proc_info_features+0x368>
  402c76:	00 00 00 
  402c79:	48 8d 05 8b 15 00 00 	lea    0x158b(%rip),%rax        # 40420b <_IO_stdin_used+0x20b>
  402c80:	48 89 05 e9 47 00 00 	mov    %rax,0x47e9(%rip)        # 407470 <proc_info_features+0x370>
  402c87:	48 8d 05 81 15 00 00 	lea    0x1581(%rip),%rax        # 40420f <_IO_stdin_used+0x20f>
  402c8e:	48 89 05 cb 47 00 00 	mov    %rax,0x47cb(%rip)        # 407460 <proc_info_features+0x360>
  402c95:	c7 05 e1 47 00 00 24 	movl   $0x24,0x47e1(%rip)        # 407480 <proc_info_features+0x380>
  402c9c:	00 00 00 
  402c9f:	48 8d 05 6d 15 00 00 	lea    0x156d(%rip),%rax        # 404213 <_IO_stdin_used+0x213>
  402ca6:	48 89 05 db 47 00 00 	mov    %rax,0x47db(%rip)        # 407488 <proc_info_features+0x388>
  402cad:	48 8d 05 63 15 00 00 	lea    0x1563(%rip),%rax        # 404217 <_IO_stdin_used+0x217>
  402cb4:	48 89 05 bd 47 00 00 	mov    %rax,0x47bd(%rip)        # 407478 <proc_info_features+0x378>
  402cbb:	c7 05 d3 47 00 00 25 	movl   $0x25,0x47d3(%rip)        # 407498 <proc_info_features+0x398>
  402cc2:	00 00 00 
  402cc5:	48 8d 05 4f 15 00 00 	lea    0x154f(%rip),%rax        # 40421b <_IO_stdin_used+0x21b>
  402ccc:	48 89 05 cd 47 00 00 	mov    %rax,0x47cd(%rip)        # 4074a0 <proc_info_features+0x3a0>
  402cd3:	48 8d 05 4a 15 00 00 	lea    0x154a(%rip),%rax        # 404224 <_IO_stdin_used+0x224>
  402cda:	48 89 05 af 47 00 00 	mov    %rax,0x47af(%rip)        # 407490 <proc_info_features+0x390>
  402ce1:	c7 05 c5 47 00 00 26 	movl   $0x26,0x47c5(%rip)        # 4074b0 <proc_info_features+0x3b0>
  402ce8:	00 00 00 
  402ceb:	48 8d 05 3b 15 00 00 	lea    0x153b(%rip),%rax        # 40422d <_IO_stdin_used+0x22d>
  402cf2:	48 89 05 bf 47 00 00 	mov    %rax,0x47bf(%rip)        # 4074b8 <proc_info_features+0x3b8>
  402cf9:	48 8d 05 36 15 00 00 	lea    0x1536(%rip),%rax        # 404236 <_IO_stdin_used+0x236>
  402d00:	48 89 05 a1 47 00 00 	mov    %rax,0x47a1(%rip)        # 4074a8 <proc_info_features+0x3a8>
  402d07:	c7 05 b7 47 00 00 27 	movl   $0x27,0x47b7(%rip)        # 4074c8 <proc_info_features+0x3c8>
  402d0e:	00 00 00 
  402d11:	48 8d 05 27 15 00 00 	lea    0x1527(%rip),%rax        # 40423f <_IO_stdin_used+0x23f>
  402d18:	48 89 05 b1 47 00 00 	mov    %rax,0x47b1(%rip)        # 4074d0 <proc_info_features+0x3d0>
  402d1f:	48 8d 05 24 15 00 00 	lea    0x1524(%rip),%rax        # 40424a <_IO_stdin_used+0x24a>
  402d26:	48 89 05 93 47 00 00 	mov    %rax,0x4793(%rip)        # 4074c0 <proc_info_features+0x3c0>
  402d2d:	c7 05 a9 47 00 00 28 	movl   $0x28,0x47a9(%rip)        # 4074e0 <proc_info_features+0x3e0>
  402d34:	00 00 00 
  402d37:	48 8d 05 17 15 00 00 	lea    0x1517(%rip),%rax        # 404255 <_IO_stdin_used+0x255>
  402d3e:	48 89 05 a3 47 00 00 	mov    %rax,0x47a3(%rip)        # 4074e8 <proc_info_features+0x3e8>
  402d45:	48 8d 05 16 15 00 00 	lea    0x1516(%rip),%rax        # 404262 <_IO_stdin_used+0x262>
  402d4c:	48 89 05 85 47 00 00 	mov    %rax,0x4785(%rip)        # 4074d8 <proc_info_features+0x3d8>
  402d53:	c7 05 9b 47 00 00 29 	movl   $0x29,0x479b(%rip)        # 4074f8 <proc_info_features+0x3f8>
  402d5a:	00 00 00 
  402d5d:	48 8d 05 0c 15 00 00 	lea    0x150c(%rip),%rax        # 404270 <_IO_stdin_used+0x270>
  402d64:	48 89 05 95 47 00 00 	mov    %rax,0x4795(%rip)        # 407500 <proc_info_features+0x400>
  402d6b:	48 8d 05 0b 15 00 00 	lea    0x150b(%rip),%rax        # 40427d <_IO_stdin_used+0x27d>
  402d72:	48 89 05 77 47 00 00 	mov    %rax,0x4777(%rip)        # 4074f0 <proc_info_features+0x3f0>
  402d79:	c7 05 8d 47 00 00 2a 	movl   $0x2a,0x478d(%rip)        # 407510 <proc_info_features+0x410>
  402d80:	00 00 00 
  402d83:	48 8d 05 01 15 00 00 	lea    0x1501(%rip),%rax        # 40428b <_IO_stdin_used+0x28b>
  402d8a:	48 89 05 87 47 00 00 	mov    %rax,0x4787(%rip)        # 407518 <proc_info_features+0x418>
  402d91:	48 8d 05 03 15 00 00 	lea    0x1503(%rip),%rax        # 40429b <_IO_stdin_used+0x29b>
  402d98:	48 89 05 69 47 00 00 	mov    %rax,0x4769(%rip)        # 407508 <proc_info_features+0x408>
  402d9f:	c7 05 7f 47 00 00 2b 	movl   $0x2b,0x477f(%rip)        # 407528 <proc_info_features+0x428>
  402da6:	00 00 00 
  402da9:	48 8d 05 fc 14 00 00 	lea    0x14fc(%rip),%rax        # 4042ac <_IO_stdin_used+0x2ac>
  402db0:	48 89 05 79 47 00 00 	mov    %rax,0x4779(%rip)        # 407530 <proc_info_features+0x430>
  402db7:	48 8d 05 fb 14 00 00 	lea    0x14fb(%rip),%rax        # 4042b9 <_IO_stdin_used+0x2b9>
  402dbe:	48 89 05 5b 47 00 00 	mov    %rax,0x475b(%rip)        # 407520 <proc_info_features+0x420>
  402dc5:	c7 05 71 47 00 00 2c 	movl   $0x2c,0x4771(%rip)        # 407540 <proc_info_features+0x440>
  402dcc:	00 00 00 
  402dcf:	48 8d 05 f1 14 00 00 	lea    0x14f1(%rip),%rax        # 4042c7 <_IO_stdin_used+0x2c7>
  402dd6:	48 89 05 6b 47 00 00 	mov    %rax,0x476b(%rip)        # 407548 <proc_info_features+0x448>
  402ddd:	48 8d 05 ef 14 00 00 	lea    0x14ef(%rip),%rax        # 4042d3 <_IO_stdin_used+0x2d3>
  402de4:	48 89 05 4d 47 00 00 	mov    %rax,0x474d(%rip)        # 407538 <proc_info_features+0x438>
  402deb:	c7 05 63 47 00 00 2d 	movl   $0x2d,0x4763(%rip)        # 407558 <proc_info_features+0x458>
  402df2:	00 00 00 
  402df5:	48 8d 05 e4 14 00 00 	lea    0x14e4(%rip),%rax        # 4042e0 <_IO_stdin_used+0x2e0>
  402dfc:	48 89 05 5d 47 00 00 	mov    %rax,0x475d(%rip)        # 407560 <proc_info_features+0x460>
  402e03:	48 8d 05 db 14 00 00 	lea    0x14db(%rip),%rax        # 4042e5 <_IO_stdin_used+0x2e5>
  402e0a:	48 89 05 3f 47 00 00 	mov    %rax,0x473f(%rip)        # 407550 <proc_info_features+0x450>
  402e11:	c7 05 55 47 00 00 2e 	movl   $0x2e,0x4755(%rip)        # 407570 <proc_info_features+0x470>
  402e18:	00 00 00 
  402e1b:	48 8d 05 c8 14 00 00 	lea    0x14c8(%rip),%rax        # 4042ea <_IO_stdin_used+0x2ea>
  402e22:	48 89 05 4f 47 00 00 	mov    %rax,0x474f(%rip)        # 407578 <proc_info_features+0x478>
  402e29:	48 8d 05 bf 14 00 00 	lea    0x14bf(%rip),%rax        # 4042ef <_IO_stdin_used+0x2ef>
  402e30:	48 89 05 31 47 00 00 	mov    %rax,0x4731(%rip)        # 407568 <proc_info_features+0x468>
  402e37:	c7 05 47 47 00 00 2f 	movl   $0x2f,0x4747(%rip)        # 407588 <proc_info_features+0x488>
  402e3e:	00 00 00 
  402e41:	48 8d 05 ac 14 00 00 	lea    0x14ac(%rip),%rax        # 4042f4 <_IO_stdin_used+0x2f4>
  402e48:	48 89 05 41 47 00 00 	mov    %rax,0x4741(%rip)        # 407590 <proc_info_features+0x490>
  402e4f:	48 8d 05 a9 14 00 00 	lea    0x14a9(%rip),%rax        # 4042ff <_IO_stdin_used+0x2ff>
  402e56:	48 89 05 23 47 00 00 	mov    %rax,0x4723(%rip)        # 407580 <proc_info_features+0x480>
  402e5d:	c7 05 39 47 00 00 30 	movl   $0x30,0x4739(%rip)        # 4075a0 <proc_info_features+0x4a0>
  402e64:	00 00 00 
  402e67:	48 8d 05 9c 14 00 00 	lea    0x149c(%rip),%rax        # 40430a <_IO_stdin_used+0x30a>
  402e6e:	48 89 05 33 47 00 00 	mov    %rax,0x4733(%rip)        # 4075a8 <proc_info_features+0x4a8>
  402e75:	48 8d 05 99 14 00 00 	lea    0x1499(%rip),%rax        # 404315 <_IO_stdin_used+0x315>
  402e7c:	48 89 05 15 47 00 00 	mov    %rax,0x4715(%rip)        # 407598 <proc_info_features+0x498>
  402e83:	c7 05 2b 47 00 00 31 	movl   $0x31,0x472b(%rip)        # 4075b8 <proc_info_features+0x4b8>
  402e8a:	00 00 00 
  402e8d:	48 8d 05 8d 14 00 00 	lea    0x148d(%rip),%rax        # 404321 <_IO_stdin_used+0x321>
  402e94:	48 89 05 25 47 00 00 	mov    %rax,0x4725(%rip)        # 4075c0 <proc_info_features+0x4c0>
  402e9b:	48 8d 05 84 14 00 00 	lea    0x1484(%rip),%rax        # 404326 <_IO_stdin_used+0x326>
  402ea2:	48 89 05 07 47 00 00 	mov    %rax,0x4707(%rip)        # 4075b0 <proc_info_features+0x4b0>
  402ea9:	c7 05 1d 47 00 00 32 	movl   $0x32,0x471d(%rip)        # 4075d0 <proc_info_features+0x4d0>
  402eb0:	00 00 00 
  402eb3:	48 8d 05 71 14 00 00 	lea    0x1471(%rip),%rax        # 40432b <_IO_stdin_used+0x32b>
  402eba:	48 89 05 17 47 00 00 	mov    %rax,0x4717(%rip)        # 4075d8 <proc_info_features+0x4d8>
  402ec1:	48 8d 05 69 14 00 00 	lea    0x1469(%rip),%rax        # 404331 <_IO_stdin_used+0x331>
  402ec8:	48 89 05 f9 46 00 00 	mov    %rax,0x46f9(%rip)        # 4075c8 <proc_info_features+0x4c8>
  402ecf:	c7 05 0f 47 00 00 33 	movl   $0x33,0x470f(%rip)        # 4075e8 <proc_info_features+0x4e8>
  402ed6:	00 00 00 
  402ed9:	48 8d 05 57 14 00 00 	lea    0x1457(%rip),%rax        # 404337 <_IO_stdin_used+0x337>
  402ee0:	48 89 05 09 47 00 00 	mov    %rax,0x4709(%rip)        # 4075f0 <proc_info_features+0x4f0>
  402ee7:	48 8d 05 4d 14 00 00 	lea    0x144d(%rip),%rax        # 40433b <_IO_stdin_used+0x33b>
  402eee:	48 89 05 eb 46 00 00 	mov    %rax,0x46eb(%rip)        # 4075e0 <proc_info_features+0x4e0>
  402ef5:	c7 05 01 47 00 00 34 	movl   $0x34,0x4701(%rip)        # 407600 <proc_info_features+0x500>
  402efc:	00 00 00 
  402eff:	48 8d 05 39 14 00 00 	lea    0x1439(%rip),%rax        # 40433f <_IO_stdin_used+0x33f>
  402f06:	48 89 05 fb 46 00 00 	mov    %rax,0x46fb(%rip)        # 407608 <proc_info_features+0x508>
  402f0d:	48 8d 05 31 14 00 00 	lea    0x1431(%rip),%rax        # 404345 <_IO_stdin_used+0x345>
  402f14:	48 89 05 dd 46 00 00 	mov    %rax,0x46dd(%rip)        # 4075f8 <proc_info_features+0x4f8>
  402f1b:	c7 05 f3 46 00 00 35 	movl   $0x35,0x46f3(%rip)        # 407618 <proc_info_features+0x518>
  402f22:	00 00 00 
  402f25:	48 8d 05 1f 14 00 00 	lea    0x141f(%rip),%rax        # 40434b <_IO_stdin_used+0x34b>
  402f2c:	48 89 05 ed 46 00 00 	mov    %rax,0x46ed(%rip)        # 407620 <proc_info_features+0x520>
  402f33:	48 8d 05 15 14 00 00 	lea    0x1415(%rip),%rax        # 40434f <_IO_stdin_used+0x34f>
  402f3a:	48 89 05 cf 46 00 00 	mov    %rax,0x46cf(%rip)        # 407610 <proc_info_features+0x510>
  402f41:	c7 05 e5 46 00 00 36 	movl   $0x36,0x46e5(%rip)        # 407630 <proc_info_features+0x530>
  402f48:	00 00 00 
  402f4b:	48 8d 05 01 14 00 00 	lea    0x1401(%rip),%rax        # 404353 <_IO_stdin_used+0x353>
  402f52:	48 89 05 df 46 00 00 	mov    %rax,0x46df(%rip)        # 407638 <proc_info_features+0x538>
  402f59:	48 8d 05 fc 13 00 00 	lea    0x13fc(%rip),%rax        # 40435c <_IO_stdin_used+0x35c>
  402f60:	48 89 05 c1 46 00 00 	mov    %rax,0x46c1(%rip)        # 407628 <proc_info_features+0x528>
  402f67:	c7 05 d7 46 00 00 37 	movl   $0x37,0x46d7(%rip)        # 407648 <proc_info_features+0x548>
  402f6e:	00 00 00 
  402f71:	48 8d 05 ed 13 00 00 	lea    0x13ed(%rip),%rax        # 404365 <_IO_stdin_used+0x365>
  402f78:	48 89 05 d1 46 00 00 	mov    %rax,0x46d1(%rip)        # 407650 <proc_info_features+0x550>
  402f7f:	48 8d 05 e7 13 00 00 	lea    0x13e7(%rip),%rax        # 40436d <_IO_stdin_used+0x36d>
  402f86:	48 89 05 b3 46 00 00 	mov    %rax,0x46b3(%rip)        # 407640 <proc_info_features+0x540>
  402f8d:	c7 05 c9 46 00 00 38 	movl   $0x38,0x46c9(%rip)        # 407660 <proc_info_features+0x560>
  402f94:	00 00 00 
  402f97:	48 8d 05 d7 13 00 00 	lea    0x13d7(%rip),%rax        # 404375 <_IO_stdin_used+0x375>
  402f9e:	48 89 05 c3 46 00 00 	mov    %rax,0x46c3(%rip)        # 407668 <proc_info_features+0x568>
  402fa5:	48 8d 05 dc 13 00 00 	lea    0x13dc(%rip),%rax        # 404388 <_IO_stdin_used+0x388>
  402fac:	48 89 05 a5 46 00 00 	mov    %rax,0x46a5(%rip)        # 407658 <proc_info_features+0x558>
  402fb3:	c7 05 bb 46 00 00 3c 	movl   $0x3c,0x46bb(%rip)        # 407678 <proc_info_features+0x578>
  402fba:	00 00 00 
  402fbd:	48 8d 05 d8 13 00 00 	lea    0x13d8(%rip),%rax        # 40439c <_IO_stdin_used+0x39c>
  402fc4:	48 89 05 b5 46 00 00 	mov    %rax,0x46b5(%rip)        # 407680 <proc_info_features+0x580>
  402fcb:	48 8d 05 d5 13 00 00 	lea    0x13d5(%rip),%rax        # 4043a7 <_IO_stdin_used+0x3a7>
  402fd2:	48 89 05 97 46 00 00 	mov    %rax,0x4697(%rip)        # 407670 <proc_info_features+0x570>
  402fd9:	c7 05 ad 46 00 00 40 	movl   $0x40,0x46ad(%rip)        # 407690 <proc_info_features+0x590>
  402fe0:	00 00 00 
  402fe3:	48 8d 05 c9 13 00 00 	lea    0x13c9(%rip),%rax        # 4043b3 <_IO_stdin_used+0x3b3>
  402fea:	48 89 05 a7 46 00 00 	mov    %rax,0x46a7(%rip)        # 407698 <proc_info_features+0x598>
  402ff1:	48 8d 05 c4 13 00 00 	lea    0x13c4(%rip),%rax        # 4043bc <_IO_stdin_used+0x3bc>
  402ff8:	48 89 05 89 46 00 00 	mov    %rax,0x4689(%rip)        # 407688 <proc_info_features+0x588>
  402fff:	c7 05 9f 46 00 00 41 	movl   $0x41,0x469f(%rip)        # 4076a8 <proc_info_features+0x5a8>
  403006:	00 00 00 
  403009:	48 8d 05 b5 13 00 00 	lea    0x13b5(%rip),%rax        # 4043c5 <_IO_stdin_used+0x3c5>
  403010:	48 89 05 99 46 00 00 	mov    %rax,0x4699(%rip)        # 4076b0 <proc_info_features+0x5b0>
  403017:	48 8d 05 af 13 00 00 	lea    0x13af(%rip),%rax        # 4043cd <_IO_stdin_used+0x3cd>
  40301e:	48 89 05 7b 46 00 00 	mov    %rax,0x467b(%rip)        # 4076a0 <proc_info_features+0x5a0>
  403025:	c7 05 91 46 00 00 42 	movl   $0x42,0x4691(%rip)        # 4076c0 <proc_info_features+0x5c0>
  40302c:	00 00 00 
  40302f:	48 8d 05 9f 13 00 00 	lea    0x139f(%rip),%rax        # 4043d5 <_IO_stdin_used+0x3d5>
  403036:	48 89 05 8b 46 00 00 	mov    %rax,0x468b(%rip)        # 4076c8 <proc_info_features+0x5c8>
  40303d:	48 8d 05 9b 13 00 00 	lea    0x139b(%rip),%rax        # 4043df <_IO_stdin_used+0x3df>
  403044:	48 89 05 6d 46 00 00 	mov    %rax,0x466d(%rip)        # 4076b8 <proc_info_features+0x5b8>
  40304b:	c7 05 83 46 00 00 43 	movl   $0x43,0x4683(%rip)        # 4076d8 <proc_info_features+0x5d8>
  403052:	00 00 00 
  403055:	48 8d 05 8d 13 00 00 	lea    0x138d(%rip),%rax        # 4043e9 <_IO_stdin_used+0x3e9>
  40305c:	48 89 05 7d 46 00 00 	mov    %rax,0x467d(%rip)        # 4076e0 <proc_info_features+0x5e0>
  403063:	48 8d 05 87 13 00 00 	lea    0x1387(%rip),%rax        # 4043f1 <_IO_stdin_used+0x3f1>
  40306a:	48 89 05 5f 46 00 00 	mov    %rax,0x465f(%rip)        # 4076d0 <proc_info_features+0x5d0>
  403071:	c7 05 75 46 00 00 44 	movl   $0x44,0x4675(%rip)        # 4076f0 <proc_info_features+0x5f0>
  403078:	00 00 00 
  40307b:	48 8d 05 77 13 00 00 	lea    0x1377(%rip),%rax        # 4043f9 <_IO_stdin_used+0x3f9>
  403082:	48 89 05 6f 46 00 00 	mov    %rax,0x466f(%rip)        # 4076f8 <proc_info_features+0x5f8>
  403089:	48 8d 05 74 13 00 00 	lea    0x1374(%rip),%rax        # 404404 <_IO_stdin_used+0x404>
  403090:	48 89 05 51 46 00 00 	mov    %rax,0x4651(%rip)        # 4076e8 <proc_info_features+0x5e8>
  403097:	c7 05 67 46 00 00 45 	movl   $0x45,0x4667(%rip)        # 407708 <proc_info_features+0x608>
  40309e:	00 00 00 
  4030a1:	48 8d 05 68 13 00 00 	lea    0x1368(%rip),%rax        # 404410 <_IO_stdin_used+0x410>
  4030a8:	48 89 05 61 46 00 00 	mov    %rax,0x4661(%rip)        # 407710 <proc_info_features+0x610>
  4030af:	48 8d 05 61 13 00 00 	lea    0x1361(%rip),%rax        # 404417 <_IO_stdin_used+0x417>
  4030b6:	48 89 05 43 46 00 00 	mov    %rax,0x4643(%rip)        # 407700 <proc_info_features+0x600>
  4030bd:	c7 05 59 46 00 00 46 	movl   $0x46,0x4659(%rip)        # 407720 <proc_info_features+0x620>
  4030c4:	00 00 00 
  4030c7:	48 8d 05 50 13 00 00 	lea    0x1350(%rip),%rax        # 40441e <_IO_stdin_used+0x41e>
  4030ce:	48 89 05 53 46 00 00 	mov    %rax,0x4653(%rip)        # 407728 <proc_info_features+0x628>
  4030d5:	48 8d 05 4a 13 00 00 	lea    0x134a(%rip),%rax        # 404426 <_IO_stdin_used+0x426>
  4030dc:	48 89 05 35 46 00 00 	mov    %rax,0x4635(%rip)        # 407718 <proc_info_features+0x618>
  4030e3:	c7 05 4b 46 00 00 47 	movl   $0x47,0x464b(%rip)        # 407738 <proc_info_features+0x638>
  4030ea:	00 00 00 
  4030ed:	48 8d 05 3b 13 00 00 	lea    0x133b(%rip),%rax        # 40442f <_IO_stdin_used+0x42f>
  4030f4:	48 89 05 45 46 00 00 	mov    %rax,0x4645(%rip)        # 407740 <proc_info_features+0x640>
  4030fb:	48 8d 05 36 13 00 00 	lea    0x1336(%rip),%rax        # 404438 <_IO_stdin_used+0x438>
  403102:	48 89 05 27 46 00 00 	mov    %rax,0x4627(%rip)        # 407730 <proc_info_features+0x630>
  403109:	c7 05 3d 46 00 00 48 	movl   $0x48,0x463d(%rip)        # 407750 <proc_info_features+0x650>
  403110:	00 00 00 
  403113:	48 8d 05 27 13 00 00 	lea    0x1327(%rip),%rax        # 404441 <_IO_stdin_used+0x441>
  40311a:	48 89 05 37 46 00 00 	mov    %rax,0x4637(%rip)        # 407758 <proc_info_features+0x658>
  403121:	48 8d 05 22 13 00 00 	lea    0x1322(%rip),%rax        # 40444a <_IO_stdin_used+0x44a>
  403128:	48 89 05 19 46 00 00 	mov    %rax,0x4619(%rip)        # 407748 <proc_info_features+0x648>
  40312f:	c7 05 2f 46 00 00 49 	movl   $0x49,0x462f(%rip)        # 407768 <proc_info_features+0x668>
  403136:	00 00 00 
  403139:	48 8d 05 13 13 00 00 	lea    0x1313(%rip),%rax        # 404453 <_IO_stdin_used+0x453>
  403140:	48 89 05 29 46 00 00 	mov    %rax,0x4629(%rip)        # 407770 <proc_info_features+0x670>
  403147:	48 8d 05 0e 13 00 00 	lea    0x130e(%rip),%rax        # 40445c <_IO_stdin_used+0x45c>
  40314e:	48 89 05 0b 46 00 00 	mov    %rax,0x460b(%rip)        # 407760 <proc_info_features+0x660>
  403155:	c7 05 21 46 00 00 4a 	movl   $0x4a,0x4621(%rip)        # 407780 <proc_info_features+0x680>
  40315c:	00 00 00 
  40315f:	48 8d 05 04 13 00 00 	lea    0x1304(%rip),%rax        # 40446a <_IO_stdin_used+0x46a>
  403166:	48 89 05 1b 46 00 00 	mov    %rax,0x461b(%rip)        # 407788 <proc_info_features+0x688>
  40316d:	48 8d 05 fe 12 00 00 	lea    0x12fe(%rip),%rax        # 404472 <_IO_stdin_used+0x472>
  403174:	48 89 05 fd 45 00 00 	mov    %rax,0x45fd(%rip)        # 407778 <proc_info_features+0x678>
  40317b:	c7 05 13 46 00 00 4b 	movl   $0x4b,0x4613(%rip)        # 407798 <proc_info_features+0x698>
  403182:	00 00 00 
  403185:	48 8d 05 d9 12 00 00 	lea    0x12d9(%rip),%rax        # 404465 <_IO_stdin_used+0x465>
  40318c:	48 89 05 0d 46 00 00 	mov    %rax,0x460d(%rip)        # 4077a0 <proc_info_features+0x6a0>
  403193:	48 8d 05 d3 12 00 00 	lea    0x12d3(%rip),%rax        # 40446d <_IO_stdin_used+0x46d>
  40319a:	48 89 05 ef 45 00 00 	mov    %rax,0x45ef(%rip)        # 407790 <proc_info_features+0x690>
  4031a1:	c6 05 48 3f 00 00 01 	movb   $0x1,0x3f48(%rip)        # 4070f0 <__libirc_isa_info_initialized>
  4031a8:	59                   	pop    %rcx
  4031a9:	c3                   	ret
  4031aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004031b0 <__libirc_get_feature_bitpos>:
  4031b0:	f3 0f 1e fa          	endbr64
  4031b4:	51                   	push   %rcx
  4031b5:	89 c1                	mov    %eax,%ecx
  4031b7:	80 3d 32 3f 00 00 00 	cmpb   $0x0,0x3f32(%rip)        # 4070f0 <__libirc_isa_info_initialized>
  4031be:	75 05                	jne    4031c5 <__libirc_get_feature_bitpos+0x15>
  4031c0:	e8 2b f5 ff ff       	call   4026f0 <__libirc_isa_init_once>
  4031c5:	89 c8                	mov    %ecx,%eax
  4031c7:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  4031cb:	48 8d 0d 2e 3f 00 00 	lea    0x3f2e(%rip),%rcx        # 407100 <proc_info_features>
  4031d2:	8b 4c c1 08          	mov    0x8(%rcx,%rax,8),%ecx
  4031d6:	8d 41 80             	lea    -0x80(%rcx),%eax
  4031d9:	3d 7f ff ff ff       	cmp    $0xffffff7f,%eax
  4031de:	b8 fd ff ff ff       	mov    $0xfffffffd,%eax
  4031e3:	0f 43 c1             	cmovae %ecx,%eax
  4031e6:	59                   	pop    %rcx
  4031e7:	c3                   	ret
  4031e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4031ef:	00 

00000000004031f0 <__libirc_get_cpu_feature>:
  4031f0:	f3 0f 1e fa          	endbr64
  4031f4:	50                   	push   %rax
  4031f5:	80 3d f4 3e 00 00 00 	cmpb   $0x0,0x3ef4(%rip)        # 4070f0 <__libirc_isa_info_initialized>
  4031fc:	75 05                	jne    403203 <__libirc_get_cpu_feature+0x13>
  4031fe:	e8 ed f4 ff ff       	call   4026f0 <__libirc_isa_init_once>
  403203:	89 f0                	mov    %esi,%eax
  403205:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  403209:	48 8d 0d f0 3e 00 00 	lea    0x3ef0(%rip),%rcx        # 407100 <proc_info_features>
  403210:	8b 4c c1 08          	mov    0x8(%rcx,%rax,8),%ecx
  403214:	8d 41 80             	lea    -0x80(%rcx),%eax
  403217:	3d 7f ff ff ff       	cmp    $0xffffff7f,%eax
  40321c:	b8 fd ff ff ff       	mov    $0xfffffffd,%eax
  403221:	0f 43 c1             	cmovae %ecx,%eax
  403224:	85 c0                	test   %eax,%eax
  403226:	78 14                	js     40323c <__libirc_get_cpu_feature+0x4c>
  403228:	89 c1                	mov    %eax,%ecx
  40322a:	c1 e9 06             	shr    $0x6,%ecx
  40322d:	48 8b 0c cf          	mov    (%rdi,%rcx,8),%rcx
  403231:	31 d2                	xor    %edx,%edx
  403233:	48 0f a3 c1          	bt     %rax,%rcx
  403237:	0f 92 c2             	setb   %dl
  40323a:	89 d0                	mov    %edx,%eax
  40323c:	59                   	pop    %rcx
  40323d:	c3                   	ret
  40323e:	66 90                	xchg   %ax,%ax

0000000000403240 <__libirc_set_cpu_feature>:
  403240:	52                   	push   %rdx
  403241:	56                   	push   %rsi
  403242:	57                   	push   %rdi
  403243:	48 89 c2             	mov    %rax,%rdx
  403246:	80 3d a3 3e 00 00 00 	cmpb   $0x0,0x3ea3(%rip)        # 4070f0 <__libirc_isa_info_initialized>
  40324d:	75 05                	jne    403254 <__libirc_set_cpu_feature+0x14>
  40324f:	e8 9c f4 ff ff       	call   4026f0 <__libirc_isa_init_once>
  403254:	89 c8                	mov    %ecx,%eax
  403256:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  40325a:	48 8d 0d 9f 3e 00 00 	lea    0x3e9f(%rip),%rcx        # 407100 <proc_info_features>
  403261:	8b 4c c1 08          	mov    0x8(%rcx,%rax,8),%ecx
  403265:	8d 41 80             	lea    -0x80(%rcx),%eax
  403268:	3d 7f ff ff ff       	cmp    $0xffffff7f,%eax
  40326d:	b8 fd ff ff ff       	mov    $0xfffffffd,%eax
  403272:	0f 43 c1             	cmovae %ecx,%eax
  403275:	85 c0                	test   %eax,%eax
  403277:	78 18                	js     403291 <__libirc_set_cpu_feature+0x51>
  403279:	89 c6                	mov    %eax,%esi
  40327b:	c1 ee 06             	shr    $0x6,%esi
  40327e:	83 e0 3f             	and    $0x3f,%eax
  403281:	bf 01 00 00 00       	mov    $0x1,%edi
  403286:	89 c1                	mov    %eax,%ecx
  403288:	48 d3 e7             	shl    %cl,%rdi
  40328b:	48 09 3c f2          	or     %rdi,(%rdx,%rsi,8)
  40328f:	31 c0                	xor    %eax,%eax
  403291:	5f                   	pop    %rdi
  403292:	5e                   	pop    %rsi
  403293:	5a                   	pop    %rdx
  403294:	c3                   	ret
  403295:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  40329c:	00 00 00 
  40329f:	90                   	nop

00000000004032a0 <__libirc_handle_intel_isa_disable>:
  4032a0:	55                   	push   %rbp
  4032a1:	41 57                	push   %r15
  4032a3:	41 56                	push   %r14
  4032a5:	41 54                	push   %r12
  4032a7:	53                   	push   %rbx
  4032a8:	31 db                	xor    %ebx,%ebx
  4032aa:	48 85 ff             	test   %rdi,%rdi
  4032ad:	0f 84 4b 01 00 00    	je     4033fe <__libirc_handle_intel_isa_disable+0x15e>
  4032b3:	49 89 fe             	mov    %rdi,%r14
  4032b6:	48 8d 3d da 0d 00 00 	lea    0xdda(%rip),%rdi        # 404097 <_IO_stdin_used+0x97>
  4032bd:	e8 6e dd ff ff       	call   401030 <getenv@plt>
  4032c2:	48 85 c0             	test   %rax,%rax
  4032c5:	0f 84 33 01 00 00    	je     4033fe <__libirc_handle_intel_isa_disable+0x15e>
  4032cb:	48 89 c2             	mov    %rax,%rdx
  4032ce:	0f b6 00             	movzbl (%rax),%eax
  4032d1:	84 c0                	test   %al,%al
  4032d3:	0f 84 25 01 00 00    	je     4033fe <__libirc_handle_intel_isa_disable+0x15e>
  4032d9:	31 db                	xor    %ebx,%ebx
  4032db:	48 8d 35 1e 3e 00 00 	lea    0x3e1e(%rip),%rsi        # 407100 <proc_info_features>
  4032e2:	31 ff                	xor    %edi,%edi
  4032e4:	4c 8d 42 01          	lea    0x1(%rdx),%r8
  4032e8:	41 b9 01 00 00 00    	mov    $0x1,%r9d
  4032ee:	49 29 d1             	sub    %rdx,%r9
  4032f1:	49 89 d2             	mov    %rdx,%r10
  4032f4:	3c 2c                	cmp    $0x2c,%al
  4032f6:	75 1a                	jne    403312 <__libirc_handle_intel_isa_disable+0x72>
  4032f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4032ff:	00 
  403300:	41 0f b6 42 01       	movzbl 0x1(%r10),%eax
  403305:	49 ff c2             	inc    %r10
  403308:	49 ff c0             	inc    %r8
  40330b:	49 ff c9             	dec    %r9
  40330e:	3c 2c                	cmp    $0x2c,%al
  403310:	74 ee                	je     403300 <__libirc_handle_intel_isa_disable+0x60>
  403312:	0f b6 c0             	movzbl %al,%eax
  403315:	85 c0                	test   %eax,%eax
  403317:	0f 84 e1 00 00 00    	je     4033fe <__libirc_handle_intel_isa_disable+0x15e>
  40331d:	4c 89 c2             	mov    %r8,%rdx
  403320:	48 89 d0             	mov    %rdx,%rax
  403323:	0f b6 0a             	movzbl (%rdx),%ecx
  403326:	48 ff c2             	inc    %rdx
  403329:	83 f9 2c             	cmp    $0x2c,%ecx
  40332c:	74 12                	je     403340 <__libirc_handle_intel_isa_disable+0xa0>
  40332e:	85 c9                	test   %ecx,%ecx
  403330:	74 0e                	je     403340 <__libirc_handle_intel_isa_disable+0xa0>
  403332:	48 89 c7             	mov    %rax,%rdi
  403335:	eb e9                	jmp    403320 <__libirc_handle_intel_isa_disable+0x80>
  403337:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40333e:	00 00 
  403340:	49 89 fb             	mov    %rdi,%r11
  403343:	4d 29 d3             	sub    %r10,%r11
  403346:	48 ff ca             	dec    %rdx
  403349:	49 ff c3             	inc    %r11
  40334c:	75 0c                	jne    40335a <__libirc_handle_intel_isa_disable+0xba>
  40334e:	0f b6 02             	movzbl (%rdx),%eax
  403351:	84 c0                	test   %al,%al
  403353:	75 8f                	jne    4032e4 <__libirc_handle_intel_isa_disable+0x44>
  403355:	e9 a4 00 00 00       	jmp    4033fe <__libirc_handle_intel_isa_disable+0x15e>
  40335a:	80 3d 8f 3d 00 00 00 	cmpb   $0x0,0x3d8f(%rip)        # 4070f0 <__libirc_isa_info_initialized>
  403361:	75 05                	jne    403368 <__libirc_handle_intel_isa_disable+0xc8>
  403363:	e8 88 f3 ff ff       	call   4026f0 <__libirc_isa_init_once>
  403368:	4c 89 d8             	mov    %r11,%rax
  40336b:	48 83 e0 fe          	and    $0xfffffffffffffffe,%rax
  40336f:	49 01 f9             	add    %rdi,%r9
  403372:	49 d1 e9             	shr    %r9
  403375:	b9 01 00 00 00       	mov    $0x1,%ecx
  40337a:	eb 14                	jmp    403390 <__libirc_handle_intel_isa_disable+0xf0>
  40337c:	0f 1f 40 00          	nopl   0x0(%rax)
  403380:	43 80 3c 1f 00       	cmpb   $0x0,(%r15,%r11,1)
  403385:	74 5b                	je     4033e2 <__libirc_handle_intel_isa_disable+0x142>
  403387:	48 ff c1             	inc    %rcx
  40338a:	48 83 f9 47          	cmp    $0x47,%rcx
  40338e:	74 be                	je     40334e <__libirc_handle_intel_isa_disable+0xae>
  403390:	4c 8d 3c 49          	lea    (%rcx,%rcx,2),%r15
  403394:	4e 8b 7c fe 10       	mov    0x10(%rsi,%r15,8),%r15
  403399:	4d 85 ff             	test   %r15,%r15
  40339c:	74 e9                	je     403387 <__libirc_handle_intel_isa_disable+0xe7>
  40339e:	49 83 fb 02          	cmp    $0x2,%r11
  4033a2:	72 2c                	jb     4033d0 <__libirc_handle_intel_isa_disable+0x130>
  4033a4:	45 31 e4             	xor    %r12d,%r12d
  4033a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4033ae:	00 00 
  4033b0:	43 0f b6 6c 60 ff    	movzbl -0x1(%r8,%r12,2),%ebp
  4033b6:	43 3a 2c 67          	cmp    (%r15,%r12,2),%bpl
  4033ba:	75 cb                	jne    403387 <__libirc_handle_intel_isa_disable+0xe7>
  4033bc:	43 0f b6 2c 60       	movzbl (%r8,%r12,2),%ebp
  4033c1:	43 3a 6c 67 01       	cmp    0x1(%r15,%r12,2),%bpl
  4033c6:	75 bf                	jne    403387 <__libirc_handle_intel_isa_disable+0xe7>
  4033c8:	49 ff c4             	inc    %r12
  4033cb:	4d 39 e1             	cmp    %r12,%r9
  4033ce:	75 e0                	jne    4033b0 <__libirc_handle_intel_isa_disable+0x110>
  4033d0:	4c 39 d8             	cmp    %r11,%rax
  4033d3:	73 ab                	jae    403380 <__libirc_handle_intel_isa_disable+0xe0>
  4033d5:	41 0f b6 2c 02       	movzbl (%r10,%rax,1),%ebp
  4033da:	41 3a 2c 07          	cmp    (%r15,%rax,1),%bpl
  4033de:	74 a0                	je     403380 <__libirc_handle_intel_isa_disable+0xe0>
  4033e0:	eb a5                	jmp    403387 <__libirc_handle_intel_isa_disable+0xe7>
  4033e2:	83 f9 02             	cmp    $0x2,%ecx
  4033e5:	0f 82 63 ff ff ff    	jb     40334e <__libirc_handle_intel_isa_disable+0xae>
  4033eb:	4c 89 f0             	mov    %r14,%rax
  4033ee:	e8 4d fe ff ff       	call   403240 <__libirc_set_cpu_feature>
  4033f3:	83 f8 01             	cmp    $0x1,%eax
  4033f6:	83 d3 00             	adc    $0x0,%ebx
  4033f9:	e9 50 ff ff ff       	jmp    40334e <__libirc_handle_intel_isa_disable+0xae>
  4033fe:	89 d8                	mov    %ebx,%eax
  403400:	5b                   	pop    %rbx
  403401:	41 5c                	pop    %r12
  403403:	41 5e                	pop    %r14
  403405:	41 5f                	pop    %r15
  403407:	5d                   	pop    %rbp
  403408:	c3                   	ret
  403409:	0f 1f 00             	nopl   (%rax)
  40340c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000403410 <__libirc_get_msg>:
  403410:	f3 0f 1e fa          	endbr64
  403414:	53                   	push   %rbx
  403415:	48 81 ec d0 00 00 00 	sub    $0xd0,%rsp
  40341c:	89 f3                	mov    %esi,%ebx
  40341e:	48 89 54 24 30       	mov    %rdx,0x30(%rsp)
  403423:	48 89 4c 24 38       	mov    %rcx,0x38(%rsp)
  403428:	4c 89 44 24 40       	mov    %r8,0x40(%rsp)
  40342d:	4c 89 4c 24 48       	mov    %r9,0x48(%rsp)
  403432:	84 c0                	test   %al,%al
  403434:	74 37                	je     40346d <__libirc_get_msg+0x5d>
  403436:	0f 29 44 24 50       	movaps %xmm0,0x50(%rsp)
  40343b:	0f 29 4c 24 60       	movaps %xmm1,0x60(%rsp)
  403440:	0f 29 54 24 70       	movaps %xmm2,0x70(%rsp)
  403445:	0f 29 9c 24 80 00 00 	movaps %xmm3,0x80(%rsp)
  40344c:	00 
  40344d:	0f 29 a4 24 90 00 00 	movaps %xmm4,0x90(%rsp)
  403454:	00 
  403455:	0f 29 ac 24 a0 00 00 	movaps %xmm5,0xa0(%rsp)
  40345c:	00 
  40345d:	0f 29 b4 24 b0 00 00 	movaps %xmm6,0xb0(%rsp)
  403464:	00 
  403465:	0f 29 bc 24 c0 00 00 	movaps %xmm7,0xc0(%rsp)
  40346c:	00 
  40346d:	e8 5e 00 00 00       	call   4034d0 <irc_ptr_msg>
  403472:	85 db                	test   %ebx,%ebx
  403474:	7e 4c                	jle    4034c2 <__libirc_get_msg+0xb2>
  403476:	48 8d 4c 24 20       	lea    0x20(%rsp),%rcx
  40347b:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
  403480:	48 8d 8c 24 e0 00 00 	lea    0xe0(%rsp),%rcx
  403487:	00 
  403488:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  40348d:	48 b9 10 00 00 00 30 	movabs $0x3000000010,%rcx
  403494:	00 00 00 
  403497:	48 89 0c 24          	mov    %rcx,(%rsp)
  40349b:	48 8d 1d 0e 43 00 00 	lea    0x430e(%rip),%rbx        # 4077b0 <get_msg_buf>
  4034a2:	49 89 e1             	mov    %rsp,%r9
  4034a5:	be 00 02 00 00       	mov    $0x200,%esi
  4034aa:	b9 00 02 00 00       	mov    $0x200,%ecx
  4034af:	48 89 df             	mov    %rbx,%rdi
  4034b2:	ba 01 00 00 00       	mov    $0x1,%edx
  4034b7:	49 89 c0             	mov    %rax,%r8
  4034ba:	e8 41 dc ff ff       	call   401100 <__vsnprintf_chk@plt>
  4034bf:	48 89 d8             	mov    %rbx,%rax
  4034c2:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  4034c9:	5b                   	pop    %rbx
  4034ca:	c3                   	ret
  4034cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004034d0 <irc_ptr_msg>:
  4034d0:	41 57                	push   %r15
  4034d2:	41 56                	push   %r14
  4034d4:	41 54                	push   %r12
  4034d6:	53                   	push   %rbx
  4034d7:	48 81 ec 88 00 00 00 	sub    $0x88,%rsp
  4034de:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4034e5:	00 00 
  4034e7:	48 89 84 24 80 00 00 	mov    %rax,0x80(%rsp)
  4034ee:	00 
  4034ef:	85 ff                	test   %edi,%edi
  4034f1:	74 37                	je     40352a <irc_ptr_msg+0x5a>
  4034f3:	89 fb                	mov    %edi,%ebx
  4034f5:	80 3d b4 46 00 00 00 	cmpb   $0x0,0x46b4(%rip)        # 407bb0 <first_msg>
  4034fc:	74 38                	je     403536 <irc_ptr_msg+0x66>
  4034fe:	48 63 c3             	movslq %ebx,%rax
  403501:	48 c1 e0 04          	shl    $0x4,%rax
  403505:	48 8d 0d c4 34 00 00 	lea    0x34c4(%rip),%rcx        # 4069d0 <irc_msgtab>
  40350c:	48 8b 44 08 08       	mov    0x8(%rax,%rcx,1),%rax
  403511:	80 3d 9c 46 00 00 01 	cmpb   $0x1,0x469c(%rip)        # 407bb4 <use_internal_msg>
  403518:	0f 85 04 01 00 00    	jne    403622 <irc_ptr_msg+0x152>
  40351e:	48 8b 3d 93 46 00 00 	mov    0x4693(%rip),%rdi        # 407bb8 <message_catalog>
  403525:	e9 e9 00 00 00       	jmp    403613 <irc_ptr_msg+0x143>
  40352a:	48 8d 05 47 0f 00 00 	lea    0xf47(%rip),%rax        # 404478 <_IO_stdin_used+0x478>
  403531:	e9 ec 00 00 00       	jmp    403622 <irc_ptr_msg+0x152>
  403536:	c6 05 73 46 00 00 01 	movb   $0x1,0x4673(%rip)        # 407bb0 <first_msg>
  40353d:	48 8d 3d 35 0f 00 00 	lea    0xf35(%rip),%rdi        # 404479 <_IO_stdin_used+0x479>
  403544:	31 f6                	xor    %esi,%esi
  403546:	e8 a5 db ff ff       	call   4010f0 <catopen@plt>
  40354b:	48 89 c7             	mov    %rax,%rdi
  40354e:	48 89 05 63 46 00 00 	mov    %rax,0x4663(%rip)        # 407bb8 <message_catalog>
  403555:	48 83 f8 ff          	cmp    $0xffffffffffffffff,%rax
  403559:	0f 85 9a 00 00 00    	jne    4035f9 <irc_ptr_msg+0x129>
  40355f:	48 8d 3d 1f 0f 00 00 	lea    0xf1f(%rip),%rdi        # 404485 <_IO_stdin_used+0x485>
  403566:	e8 c5 da ff ff       	call   401030 <getenv@plt>
  40356b:	48 85 c0             	test   %rax,%rax
  40356e:	74 78                	je     4035e8 <irc_ptr_msg+0x118>
  403570:	49 89 e6             	mov    %rsp,%r14
  403573:	ba 80 00 00 00       	mov    $0x80,%edx
  403578:	b9 80 00 00 00       	mov    $0x80,%ecx
  40357d:	4c 89 f7             	mov    %r14,%rdi
  403580:	48 89 c6             	mov    %rax,%rsi
  403583:	e8 88 db ff ff       	call   401110 <__strncpy_chk@plt>
  403588:	c6 44 24 7f 00       	movb   $0x0,0x7f(%rsp)
  40358d:	4c 89 f7             	mov    %r14,%rdi
  403590:	be 2e 00 00 00       	mov    $0x2e,%esi
  403595:	e8 06 db ff ff       	call   4010a0 <strchr@plt>
  40359a:	48 85 c0             	test   %rax,%rax
  40359d:	74 49                	je     4035e8 <irc_ptr_msg+0x118>
  40359f:	49 89 c6             	mov    %rax,%r14
  4035a2:	c6 00 00             	movb   $0x0,(%rax)
  4035a5:	4c 8d 3d d9 0e 00 00 	lea    0xed9(%rip),%r15        # 404485 <_IO_stdin_used+0x485>
  4035ac:	49 89 e4             	mov    %rsp,%r12
  4035af:	4c 89 ff             	mov    %r15,%rdi
  4035b2:	4c 89 e6             	mov    %r12,%rsi
  4035b5:	ba 01 00 00 00       	mov    $0x1,%edx
  4035ba:	e8 91 da ff ff       	call   401050 <setenv@plt>
  4035bf:	48 8d 3d b3 0e 00 00 	lea    0xeb3(%rip),%rdi        # 404479 <_IO_stdin_used+0x479>
  4035c6:	31 f6                	xor    %esi,%esi
  4035c8:	e8 23 db ff ff       	call   4010f0 <catopen@plt>
  4035cd:	48 89 05 e4 45 00 00 	mov    %rax,0x45e4(%rip)        # 407bb8 <message_catalog>
  4035d4:	41 c6 06 2e          	movb   $0x2e,(%r14)
  4035d8:	4c 89 ff             	mov    %r15,%rdi
  4035db:	4c 89 e6             	mov    %r12,%rsi
  4035de:	ba 01 00 00 00       	mov    $0x1,%edx
  4035e3:	e8 68 da ff ff       	call   401050 <setenv@plt>
  4035e8:	48 8b 3d c9 45 00 00 	mov    0x45c9(%rip),%rdi        # 407bb8 <message_catalog>
  4035ef:	48 83 ff ff          	cmp    $0xffffffffffffffff,%rdi
  4035f3:	0f 84 05 ff ff ff    	je     4034fe <irc_ptr_msg+0x2e>
  4035f9:	c6 05 b4 45 00 00 01 	movb   $0x1,0x45b4(%rip)        # 407bb4 <use_internal_msg>
  403600:	48 63 c3             	movslq %ebx,%rax
  403603:	48 c1 e0 04          	shl    $0x4,%rax
  403607:	48 8d 0d c2 33 00 00 	lea    0x33c2(%rip),%rcx        # 4069d0 <irc_msgtab>
  40360e:	48 8b 44 08 08       	mov    0x8(%rax,%rcx,1),%rax
  403613:	be 01 00 00 00       	mov    $0x1,%esi
  403618:	89 da                	mov    %ebx,%edx
  40361a:	48 89 c1             	mov    %rax,%rcx
  40361d:	e8 2e db ff ff       	call   401150 <catgets@plt>
  403622:	64 48 8b 0c 25 28 00 	mov    %fs:0x28,%rcx
  403629:	00 00 
  40362b:	48 3b 8c 24 80 00 00 	cmp    0x80(%rsp),%rcx
  403632:	00 
  403633:	75 0f                	jne    403644 <irc_ptr_msg+0x174>
  403635:	48 81 c4 88 00 00 00 	add    $0x88,%rsp
  40363c:	5b                   	pop    %rbx
  40363d:	41 5c                	pop    %r12
  40363f:	41 5e                	pop    %r14
  403641:	41 5f                	pop    %r15
  403643:	c3                   	ret
  403644:	e8 47 da ff ff       	call   401090 <__stack_chk_fail@plt>
  403649:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000403650 <__libirc_print>:
  403650:	f3 0f 1e fa          	endbr64
  403654:	55                   	push   %rbp
  403655:	41 56                	push   %r14
  403657:	53                   	push   %rbx
  403658:	48 81 ec d0 00 00 00 	sub    $0xd0,%rsp
  40365f:	89 fb                	mov    %edi,%ebx
  403661:	48 89 4c 24 38       	mov    %rcx,0x38(%rsp)
  403666:	4c 89 44 24 40       	mov    %r8,0x40(%rsp)
  40366b:	4c 89 4c 24 48       	mov    %r9,0x48(%rsp)
  403670:	84 c0                	test   %al,%al
  403672:	74 37                	je     4036ab <__libirc_print+0x5b>
  403674:	0f 29 44 24 50       	movaps %xmm0,0x50(%rsp)
  403679:	0f 29 4c 24 60       	movaps %xmm1,0x60(%rsp)
  40367e:	0f 29 54 24 70       	movaps %xmm2,0x70(%rsp)
  403683:	0f 29 9c 24 80 00 00 	movaps %xmm3,0x80(%rsp)
  40368a:	00 
  40368b:	0f 29 a4 24 90 00 00 	movaps %xmm4,0x90(%rsp)
  403692:	00 
  403693:	0f 29 ac 24 a0 00 00 	movaps %xmm5,0xa0(%rsp)
  40369a:	00 
  40369b:	0f 29 b4 24 b0 00 00 	movaps %xmm6,0xb0(%rsp)
  4036a2:	00 
  4036a3:	0f 29 bc 24 c0 00 00 	movaps %xmm7,0xc0(%rsp)
  4036aa:	00 
  4036ab:	85 f6                	test   %esi,%esi
  4036ad:	0f 84 80 00 00 00    	je     403733 <__libirc_print+0xe3>
  4036b3:	89 d5                	mov    %edx,%ebp
  4036b5:	89 f7                	mov    %esi,%edi
  4036b7:	e8 14 fe ff ff       	call   4034d0 <irc_ptr_msg>
  4036bc:	85 ed                	test   %ebp,%ebp
  4036be:	7e 4c                	jle    40370c <__libirc_print+0xbc>
  4036c0:	48 8d 4c 24 20       	lea    0x20(%rsp),%rcx
  4036c5:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
  4036ca:	48 8d 8c 24 f0 00 00 	lea    0xf0(%rsp),%rcx
  4036d1:	00 
  4036d2:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  4036d7:	48 b9 18 00 00 00 30 	movabs $0x3000000018,%rcx
  4036de:	00 00 00 
  4036e1:	48 89 0c 24          	mov    %rcx,(%rsp)
  4036e5:	4c 8d 35 c4 42 00 00 	lea    0x42c4(%rip),%r14        # 4079b0 <print_buf>
  4036ec:	49 89 e1             	mov    %rsp,%r9
  4036ef:	be 00 02 00 00       	mov    $0x200,%esi
  4036f4:	b9 00 02 00 00       	mov    $0x200,%ecx
  4036f9:	4c 89 f7             	mov    %r14,%rdi
  4036fc:	ba 01 00 00 00       	mov    $0x1,%edx
  403701:	49 89 c0             	mov    %rax,%r8
  403704:	e8 f7 d9 ff ff       	call   401100 <__vsnprintf_chk@plt>
  403709:	4c 89 f0             	mov    %r14,%rax
  40370c:	83 fb 01             	cmp    $0x1,%ebx
  40370f:	75 4f                	jne    403760 <__libirc_print+0x110>
  403711:	48 8b 0d c0 38 00 00 	mov    0x38c0(%rip),%rcx        # 406fd8 <stderr@GLIBC_2.2.5-0xe8>
  403718:	48 8b 39             	mov    (%rcx),%rdi
  40371b:	48 8d 15 53 0d 00 00 	lea    0xd53(%rip),%rdx        # 404475 <_IO_stdin_used+0x475>
  403722:	be 01 00 00 00       	mov    $0x1,%esi
  403727:	48 89 c1             	mov    %rax,%rcx
  40372a:	31 c0                	xor    %eax,%eax
  40372c:	e8 3f da ff ff       	call   401170 <__fprintf_chk@plt>
  403731:	eb 43                	jmp    403776 <__libirc_print+0x126>
  403733:	83 fb 01             	cmp    $0x1,%ebx
  403736:	75 4a                	jne    403782 <__libirc_print+0x132>
  403738:	48 8b 05 99 38 00 00 	mov    0x3899(%rip),%rax        # 406fd8 <stderr@GLIBC_2.2.5-0xe8>
  40373f:	48 8b 38             	mov    (%rax),%rdi
  403742:	48 8d 15 2e 0d 00 00 	lea    0xd2e(%rip),%rdx        # 404477 <_IO_stdin_used+0x477>
  403749:	be 01 00 00 00       	mov    $0x1,%esi
  40374e:	31 c0                	xor    %eax,%eax
  403750:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  403757:	5b                   	pop    %rbx
  403758:	41 5e                	pop    %r14
  40375a:	5d                   	pop    %rbp
  40375b:	e9 10 da ff ff       	jmp    401170 <__fprintf_chk@plt>
  403760:	48 8d 35 0e 0d 00 00 	lea    0xd0e(%rip),%rsi        # 404475 <_IO_stdin_used+0x475>
  403767:	bf 01 00 00 00       	mov    $0x1,%edi
  40376c:	48 89 c2             	mov    %rax,%rdx
  40376f:	31 c0                	xor    %eax,%eax
  403771:	e8 aa d9 ff ff       	call   401120 <__printf_chk@plt>
  403776:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  40377d:	5b                   	pop    %rbx
  40377e:	41 5e                	pop    %r14
  403780:	5d                   	pop    %rbp
  403781:	c3                   	ret
  403782:	48 8d 35 ee 0c 00 00 	lea    0xcee(%rip),%rsi        # 404477 <_IO_stdin_used+0x477>
  403789:	bf 01 00 00 00       	mov    $0x1,%edi
  40378e:	31 c0                	xor    %eax,%eax
  403790:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  403797:	5b                   	pop    %rbx
  403798:	41 5e                	pop    %r14
  40379a:	5d                   	pop    %rbp
  40379b:	e9 80 d9 ff ff       	jmp    401120 <__printf_chk@plt>

Disassembly of section .fini:

00000000004037a0 <_fini>:
  4037a0:	48 83 ec 08          	sub    $0x8,%rsp
  4037a4:	48 83 c4 08          	add    $0x8,%rsp
  4037a8:	c3                   	ret
