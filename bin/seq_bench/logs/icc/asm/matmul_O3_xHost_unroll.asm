
bin/seq_bench/icc/matmul_O3_xHost_unroll:     file format elf64-x86-64


Disassembly of section .init:

0000000000401000 <_init>:
  401000:	48 83 ec 08          	sub    $0x8,%rsp
  401004:	48 8b 05 b5 6f 00 00 	mov    0x6fb5(%rip),%rax        # 407fc0 <__gmon_start__@Base>
  40100b:	48 85 c0             	test   %rax,%rax
  40100e:	74 02                	je     401012 <_init+0x12>
  401010:	ff d0                	call   *%rax
  401012:	48 83 c4 08          	add    $0x8,%rsp
  401016:	c3                   	ret

Disassembly of section .plt:

0000000000401020 <getenv@plt-0x10>:
  401020:	ff 35 ca 6f 00 00    	push   0x6fca(%rip)        # 407ff0 <_GLOBAL_OFFSET_TABLE_+0x8>
  401026:	ff 25 cc 6f 00 00    	jmp    *0x6fcc(%rip)        # 407ff8 <_GLOBAL_OFFSET_TABLE_+0x10>
  40102c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401030 <getenv@plt>:
  401030:	ff 25 ca 6f 00 00    	jmp    *0x6fca(%rip)        # 408000 <getenv@GLIBC_2.2.5>
  401036:	68 00 00 00 00       	push   $0x0
  40103b:	e9 e0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401040 <free@plt>:
  401040:	ff 25 c2 6f 00 00    	jmp    *0x6fc2(%rip)        # 408008 <free@GLIBC_2.2.5>
  401046:	68 01 00 00 00       	push   $0x1
  40104b:	e9 d0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401050 <setenv@plt>:
  401050:	ff 25 ba 6f 00 00    	jmp    *0x6fba(%rip)        # 408010 <setenv@GLIBC_2.2.5>
  401056:	68 02 00 00 00       	push   $0x2
  40105b:	e9 c0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401060 <clock@plt>:
  401060:	ff 25 b2 6f 00 00    	jmp    *0x6fb2(%rip)        # 408018 <clock@GLIBC_2.2.5>
  401066:	68 03 00 00 00       	push   $0x3
  40106b:	e9 b0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401070 <fclose@plt>:
  401070:	ff 25 aa 6f 00 00    	jmp    *0x6faa(%rip)        # 408020 <fclose@GLIBC_2.2.5>
  401076:	68 04 00 00 00       	push   $0x4
  40107b:	e9 a0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401080 <strlen@plt>:
  401080:	ff 25 a2 6f 00 00    	jmp    *0x6fa2(%rip)        # 408028 <strlen@GLIBC_2.2.5>
  401086:	68 05 00 00 00       	push   $0x5
  40108b:	e9 90 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401090 <__stack_chk_fail@plt>:
  401090:	ff 25 9a 6f 00 00    	jmp    *0x6f9a(%rip)        # 408030 <__stack_chk_fail@GLIBC_2.4>
  401096:	68 06 00 00 00       	push   $0x6
  40109b:	e9 80 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010a0 <strchr@plt>:
  4010a0:	ff 25 92 6f 00 00    	jmp    *0x6f92(%rip)        # 408038 <strchr@GLIBC_2.2.5>
  4010a6:	68 07 00 00 00       	push   $0x7
  4010ab:	e9 70 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010b0 <memset@plt>:
  4010b0:	ff 25 8a 6f 00 00    	jmp    *0x6f8a(%rip)        # 408040 <memset@GLIBC_2.2.5>
  4010b6:	68 08 00 00 00       	push   $0x8
  4010bb:	e9 60 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010c0 <fputc@plt>:
  4010c0:	ff 25 82 6f 00 00    	jmp    *0x6f82(%rip)        # 408048 <fputc@GLIBC_2.2.5>
  4010c6:	68 09 00 00 00       	push   $0x9
  4010cb:	e9 50 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010d0 <fprintf@plt>:
  4010d0:	ff 25 7a 6f 00 00    	jmp    *0x6f7a(%rip)        # 408050 <fprintf@GLIBC_2.2.5>
  4010d6:	68 0a 00 00 00       	push   $0xa
  4010db:	e9 40 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010e0 <malloc@plt>:
  4010e0:	ff 25 72 6f 00 00    	jmp    *0x6f72(%rip)        # 408058 <malloc@GLIBC_2.2.5>
  4010e6:	68 0b 00 00 00       	push   $0xb
  4010eb:	e9 30 ff ff ff       	jmp    401020 <_init+0x20>

00000000004010f0 <catopen@plt>:
  4010f0:	ff 25 6a 6f 00 00    	jmp    *0x6f6a(%rip)        # 408060 <catopen@GLIBC_2.2.5>
  4010f6:	68 0c 00 00 00       	push   $0xc
  4010fb:	e9 20 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401100 <__vsnprintf_chk@plt>:
  401100:	ff 25 62 6f 00 00    	jmp    *0x6f62(%rip)        # 408068 <__vsnprintf_chk@GLIBC_2.3.4>
  401106:	68 0d 00 00 00       	push   $0xd
  40110b:	e9 10 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401110 <__strncpy_chk@plt>:
  401110:	ff 25 5a 6f 00 00    	jmp    *0x6f5a(%rip)        # 408070 <__strncpy_chk@GLIBC_2.3.4>
  401116:	68 0e 00 00 00       	push   $0xe
  40111b:	e9 00 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401120 <__printf_chk@plt>:
  401120:	ff 25 52 6f 00 00    	jmp    *0x6f52(%rip)        # 408078 <__printf_chk@GLIBC_2.3.4>
  401126:	68 0f 00 00 00       	push   $0xf
  40112b:	e9 f0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401130 <fopen@plt>:
  401130:	ff 25 4a 6f 00 00    	jmp    *0x6f4a(%rip)        # 408080 <fopen@GLIBC_2.2.5>
  401136:	68 10 00 00 00       	push   $0x10
  40113b:	e9 e0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401140 <perror@plt>:
  401140:	ff 25 42 6f 00 00    	jmp    *0x6f42(%rip)        # 408088 <perror@GLIBC_2.2.5>
  401146:	68 11 00 00 00       	push   $0x11
  40114b:	e9 d0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401150 <catgets@plt>:
  401150:	ff 25 3a 6f 00 00    	jmp    *0x6f3a(%rip)        # 408090 <catgets@GLIBC_2.2.5>
  401156:	68 12 00 00 00       	push   $0x12
  40115b:	e9 c0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401160 <exit@plt>:
  401160:	ff 25 32 6f 00 00    	jmp    *0x6f32(%rip)        # 408098 <exit@GLIBC_2.2.5>
  401166:	68 13 00 00 00       	push   $0x13
  40116b:	e9 b0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401170 <__fprintf_chk@plt>:
  401170:	ff 25 2a 6f 00 00    	jmp    *0x6f2a(%rip)        # 4080a0 <__fprintf_chk@GLIBC_2.3.4>
  401176:	68 14 00 00 00       	push   $0x14
  40117b:	e9 a0 fe ff ff       	jmp    401020 <_init+0x20>

0000000000401180 <__strncat_chk@plt>:
  401180:	ff 25 22 6f 00 00    	jmp    *0x6f22(%rip)        # 4080a8 <__strncat_chk@GLIBC_2.3.4>
  401186:	68 15 00 00 00       	push   $0x15
  40118b:	e9 90 fe ff ff       	jmp    401020 <_init+0x20>

Disassembly of section .plt.got:

0000000000401190 <__cxa_finalize@plt>:
  401190:	ff 25 3a 6e 00 00    	jmp    *0x6e3a(%rip)        # 407fd0 <__cxa_finalize@GLIBC_2.2.5>
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
  4011bb:	ff 15 ef 6d 00 00    	call   *0x6def(%rip)        # 407fb0 <__libc_start_main@GLIBC_2.34>
  4011c1:	f4                   	hlt
  4011c2:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4011c9:	00 00 00 
  4011cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004011d0 <deregister_tm_clones>:
  4011d0:	48 8d 3d e9 6e 00 00 	lea    0x6ee9(%rip),%rdi        # 4080c0 <stderr@GLIBC_2.2.5>
  4011d7:	48 8d 05 e2 6e 00 00 	lea    0x6ee2(%rip),%rax        # 4080c0 <stderr@GLIBC_2.2.5>
  4011de:	48 39 f8             	cmp    %rdi,%rax
  4011e1:	74 15                	je     4011f8 <deregister_tm_clones+0x28>
  4011e3:	48 8b 05 ce 6d 00 00 	mov    0x6dce(%rip),%rax        # 407fb8 <_ITM_deregisterTMCloneTable@Base>
  4011ea:	48 85 c0             	test   %rax,%rax
  4011ed:	74 09                	je     4011f8 <deregister_tm_clones+0x28>
  4011ef:	ff e0                	jmp    *%rax
  4011f1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4011f8:	c3                   	ret
  4011f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401200 <register_tm_clones>:
  401200:	48 8d 3d b9 6e 00 00 	lea    0x6eb9(%rip),%rdi        # 4080c0 <stderr@GLIBC_2.2.5>
  401207:	48 8d 35 b2 6e 00 00 	lea    0x6eb2(%rip),%rsi        # 4080c0 <stderr@GLIBC_2.2.5>
  40120e:	48 29 fe             	sub    %rdi,%rsi
  401211:	48 89 f0             	mov    %rsi,%rax
  401214:	48 c1 ee 3f          	shr    $0x3f,%rsi
  401218:	48 c1 f8 03          	sar    $0x3,%rax
  40121c:	48 01 c6             	add    %rax,%rsi
  40121f:	48 d1 fe             	sar    %rsi
  401222:	74 14                	je     401238 <register_tm_clones+0x38>
  401224:	48 8b 05 9d 6d 00 00 	mov    0x6d9d(%rip),%rax        # 407fc8 <_ITM_registerTMCloneTable@Base>
  40122b:	48 85 c0             	test   %rax,%rax
  40122e:	74 08                	je     401238 <register_tm_clones+0x38>
  401230:	ff e0                	jmp    *%rax
  401232:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401238:	c3                   	ret
  401239:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401240 <__do_global_dtors_aux>:
  401240:	f3 0f 1e fa          	endbr64
  401244:	80 3d 7d 6e 00 00 00 	cmpb   $0x0,0x6e7d(%rip)        # 4080c8 <completed.0>
  40124b:	75 2b                	jne    401278 <__do_global_dtors_aux+0x38>
  40124d:	55                   	push   %rbp
  40124e:	48 83 3d 7a 6d 00 00 	cmpq   $0x0,0x6d7a(%rip)        # 407fd0 <__cxa_finalize@GLIBC_2.2.5>
  401255:	00 
  401256:	48 89 e5             	mov    %rsp,%rbp
  401259:	74 0c                	je     401267 <__do_global_dtors_aux+0x27>
  40125b:	48 8b 3d 56 6e 00 00 	mov    0x6e56(%rip),%rdi        # 4080b8 <__dso_handle>
  401262:	e8 29 ff ff ff       	call   401190 <__cxa_finalize@plt>
  401267:	e8 64 ff ff ff       	call   4011d0 <deregister_tm_clones>
  40126c:	c6 05 55 6e 00 00 01 	movb   $0x1,0x6e55(%rip)        # 4080c8 <completed.0>
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
  4012a1:	48 81 ec d8 03 00 00 	sub    $0x3d8,%rsp
  4012a8:	bf 03 00 00 00       	mov    $0x3,%edi
  4012ad:	be fe 9f 9d 00       	mov    $0x9d9ffe,%esi
  4012b2:	e8 99 15 00 00       	call   402850 <__intel_new_feature_proc_init>
  4012b7:	c5 f8 ae 1c 24       	vstmxcsr (%rsp)
  4012bc:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4012c1:	81 0c 24 40 80 00 00 	orl    $0x8040,(%rsp)
  4012c8:	c5 f8 ae 14 24       	vldmxcsr (%rsp)
  4012cd:	e8 0e fe ff ff       	call   4010e0 <malloc@plt>
  4012d2:	49 89 c7             	mov    %rax,%r15
  4012d5:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4012da:	e8 01 fe ff ff       	call   4010e0 <malloc@plt>
  4012df:	48 89 c3             	mov    %rax,%rbx
  4012e2:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4012e7:	e8 f4 fd ff ff       	call   4010e0 <malloc@plt>
  4012ec:	49 89 c5             	mov    %rax,%r13
  4012ef:	45 33 f6             	xor    %r14d,%r14d
  4012f2:	33 d2                	xor    %edx,%edx
  4012f4:	c5 fd 10 15 24 3d 00 	vmovupd 0x3d24(%rip),%ymm2        # 405020 <_IO_stdin_used+0x20>
  4012fb:	00 
  4012fc:	48 b8 00 00 00 00 00 	movabs $0x4000000000000000,%rax
  401303:	00 00 40 
  401306:	c5 fd 10 0d 32 3d 00 	vmovupd 0x3d32(%rip),%ymm1        # 405040 <_IO_stdin_used+0x40>
  40130d:	00 
  40130e:	c5 fd 57 c0          	vxorpd %ymm0,%ymm0,%ymm0
  401312:	49 b9 00 00 00 00 00 	movabs $0x4008000000000000,%r9
  401319:	00 08 40 
  40131c:	48 89 9c 24 00 02 00 	mov    %rbx,0x200(%rsp)
  401323:	00 
  401324:	33 f6                	xor    %esi,%esi
  401326:	4c 89 bc 24 80 02 00 	mov    %r15,0x280(%rsp)
  40132d:	00 
  40132e:	48 89 d9             	mov    %rbx,%rcx
  401331:	4d 89 f8             	mov    %r15,%r8
  401334:	4d 8d 7c 35 00       	lea    0x0(%r13,%rsi,1),%r15
  401339:	4d 89 fc             	mov    %r15,%r12
  40133c:	49 83 e4 1f          	and    $0x1f,%r12
  401340:	45 85 e4             	test   %r12d,%r12d
  401343:	74 34                	je     401379 <main+0xe9>
  401345:	41 f7 c4 07 00 00 00 	test   $0x7,%r12d
  40134c:	0f 85 43 11 00 00    	jne    402495 <main+0x1205>
  401352:	41 f7 dc             	neg    %r12d
  401355:	4d 89 f2             	mov    %r14,%r10
  401358:	41 83 c4 20          	add    $0x20,%r12d
  40135c:	41 c1 ec 03          	shr    $0x3,%r12d
  401360:	44 89 e3             	mov    %r12d,%ebx
  401363:	4b 89 04 d0          	mov    %rax,(%r8,%r10,8)
  401367:	4e 89 0c d1          	mov    %r9,(%rcx,%r10,8)
  40136b:	4f 89 34 d7          	mov    %r14,(%r15,%r10,8)
  40136f:	49 ff c2             	inc    %r10
  401372:	4c 3b d3             	cmp    %rbx,%r10
  401375:	72 ec                	jb     401363 <main+0xd3>
  401377:	eb 03                	jmp    40137c <main+0xec>
  401379:	4c 89 f3             	mov    %r14,%rbx
  40137c:	41 f7 dc             	neg    %r12d
  40137f:	41 83 c4 08          	add    $0x8,%r12d
  401383:	41 83 e4 0f          	and    $0xf,%r12d
  401387:	41 f7 dc             	neg    %r12d
  40138a:	41 81 c4 88 13 00 00 	add    $0x1388,%r12d
  401391:	45 89 e2             	mov    %r12d,%r10d
  401394:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401399:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4013a0:	c4 c1 7d 11 14 d8    	vmovupd %ymm2,(%r8,%rbx,8)
  4013a6:	c5 fd 11 0c d9       	vmovupd %ymm1,(%rcx,%rbx,8)
  4013ab:	c4 c1 7d 11 04 df    	vmovupd %ymm0,(%r15,%rbx,8)
  4013b1:	c4 c1 7d 11 54 d8 20 	vmovupd %ymm2,0x20(%r8,%rbx,8)
  4013b8:	c5 fd 11 4c d9 20    	vmovupd %ymm1,0x20(%rcx,%rbx,8)
  4013be:	c4 c1 7d 11 44 df 20 	vmovupd %ymm0,0x20(%r15,%rbx,8)
  4013c5:	c4 c1 7d 11 54 d8 40 	vmovupd %ymm2,0x40(%r8,%rbx,8)
  4013cc:	c5 fd 11 4c d9 40    	vmovupd %ymm1,0x40(%rcx,%rbx,8)
  4013d2:	c4 c1 7d 11 44 df 40 	vmovupd %ymm0,0x40(%r15,%rbx,8)
  4013d9:	c4 c1 7d 11 54 d8 60 	vmovupd %ymm2,0x60(%r8,%rbx,8)
  4013e0:	c5 fd 11 4c d9 60    	vmovupd %ymm1,0x60(%rcx,%rbx,8)
  4013e6:	c4 c1 7d 11 44 df 60 	vmovupd %ymm0,0x60(%r15,%rbx,8)
  4013ed:	48 83 c3 10          	add    $0x10,%rbx
  4013f1:	49 3b da             	cmp    %r10,%rbx
  4013f4:	72 aa                	jb     4013a0 <main+0x110>
  4013f6:	41 8d 5c 24 01       	lea    0x1(%r12),%ebx
  4013fb:	81 fb 88 13 00 00    	cmp    $0x1388,%ebx
  401401:	77 60                	ja     401463 <main+0x1d3>
  401403:	45 89 e3             	mov    %r12d,%r11d
  401406:	41 f7 db             	neg    %r11d
  401409:	41 81 c3 88 13 00 00 	add    $0x1388,%r11d
  401410:	41 83 fb 04          	cmp    $0x4,%r11d
  401414:	0f 82 bd 10 00 00    	jb     4024d7 <main+0x1247>
  40141a:	45 89 da             	mov    %r11d,%r10d
  40141d:	33 db                	xor    %ebx,%ebx
  40141f:	41 83 e2 fc          	and    $0xfffffffc,%r10d
  401423:	41 8d 3c 1c          	lea    (%r12,%rbx,1),%edi
  401427:	83 c3 04             	add    $0x4,%ebx
  40142a:	48 63 ff             	movslq %edi,%rdi
  40142d:	41 3b da             	cmp    %r10d,%ebx
  401430:	c4 c1 7d 11 14 f8    	vmovupd %ymm2,(%r8,%rdi,8)
  401436:	c5 fd 11 0c f9       	vmovupd %ymm1,(%rcx,%rdi,8)
  40143b:	c4 c1 7d 11 04 ff    	vmovupd %ymm0,(%r15,%rdi,8)
  401441:	72 e0                	jb     401423 <main+0x193>
  401443:	45 3b d3             	cmp    %r11d,%r10d
  401446:	73 1b                	jae    401463 <main+0x1d3>
  401448:	43 8d 1c 14          	lea    (%r12,%r10,1),%ebx
  40144c:	41 ff c2             	inc    %r10d
  40144f:	48 63 db             	movslq %ebx,%rbx
  401452:	49 89 04 d8          	mov    %rax,(%r8,%rbx,8)
  401456:	4c 89 0c d9          	mov    %r9,(%rcx,%rbx,8)
  40145a:	4d 89 34 df          	mov    %r14,(%r15,%rbx,8)
  40145e:	45 3b d3             	cmp    %r11d,%r10d
  401461:	72 e5                	jb     401448 <main+0x1b8>
  401463:	ff c2                	inc    %edx
  401465:	48 81 c1 40 9c 00 00 	add    $0x9c40,%rcx
  40146c:	49 81 c0 40 9c 00 00 	add    $0x9c40,%r8
  401473:	48 81 c6 40 9c 00 00 	add    $0x9c40,%rsi
  40147a:	81 fa 88 13 00 00    	cmp    $0x1388,%edx
  401480:	0f 82 ae fe ff ff    	jb     401334 <main+0xa4>
  401486:	c5 f8 77             	vzeroupper
  401489:	48 8b 9c 24 00 02 00 	mov    0x200(%rsp),%rbx
  401490:	00 
  401491:	4c 8b bc 24 80 02 00 	mov    0x280(%rsp),%r15
  401498:	00 
  401499:	e8 c2 fb ff ff       	call   401060 <clock@plt>
  40149e:	49 89 c4             	mov    %rax,%r12
  4014a1:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
  4014a6:	33 f6                	xor    %esi,%esi
  4014a8:	4c 89 ac 24 98 02 00 	mov    %r13,0x298(%rsp)
  4014af:	00 
  4014b0:	41 bb 80 00 00 00    	mov    $0x80,%r11d
  4014b6:	89 f1                	mov    %esi,%ecx
  4014b8:	33 c0                	xor    %eax,%eax
  4014ba:	c1 e1 07             	shl    $0x7,%ecx
  4014bd:	f7 d9                	neg    %ecx
  4014bf:	81 c1 88 13 00 00    	add    $0x1388,%ecx
  4014c5:	81 f9 80 00 00 00    	cmp    $0x80,%ecx
  4014cb:	41 0f 43 cb          	cmovae %r11d,%ecx
  4014cf:	89 ca                	mov    %ecx,%edx
  4014d1:	c1 ea 02             	shr    $0x2,%edx
  4014d4:	89 54 24 10          	mov    %edx,0x10(%rsp)
  4014d8:	89 0c 24             	mov    %ecx,(%rsp)
  4014db:	41 89 c0             	mov    %eax,%r8d
  4014de:	89 f1                	mov    %esi,%ecx
  4014e0:	41 c1 e0 07          	shl    $0x7,%r8d
  4014e4:	4d 89 f2             	mov    %r14,%r10
  4014e7:	49 69 f8 40 9c 00 00 	imul   $0x9c40,%r8,%rdi
  4014ee:	45 89 c5             	mov    %r8d,%r13d
  4014f1:	48 03 fb             	add    %rbx,%rdi
  4014f4:	41 f7 dd             	neg    %r13d
  4014f7:	41 81 c5 88 13 00 00 	add    $0x1388,%r13d
  4014fe:	41 81 fd 80 00 00 00 	cmp    $0x80,%r13d
  401505:	89 c2                	mov    %eax,%edx
  401507:	45 0f 43 eb          	cmovae %r11d,%r13d
  40150b:	48 c1 e2 0a          	shl    $0xa,%rdx
  40150f:	45 89 ec             	mov    %r13d,%r12d
  401512:	4d 63 cd             	movslq %r13d,%r9
  401515:	49 03 d7             	add    %r15,%rdx
  401518:	c1 e1 07             	shl    $0x7,%ecx
  40151b:	41 c1 ec 02          	shr    $0x2,%r12d
  40151f:	4c 89 8c 24 28 02 00 	mov    %r9,0x228(%rsp)
  401526:	00 
  401527:	44 89 84 24 20 02 00 	mov    %r8d,0x220(%rsp)
  40152e:	00 
  40152f:	89 b4 24 88 02 00 00 	mov    %esi,0x288(%rsp)
  401536:	44 89 a4 24 b8 01 00 	mov    %r12d,0x1b8(%rsp)
  40153d:	00 
  40153e:	44 89 ac 24 b0 01 00 	mov    %r13d,0x1b0(%rsp)
  401545:	00 
  401546:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  40154b:	89 8c 24 c0 01 00 00 	mov    %ecx,0x1c0(%rsp)
  401552:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
  401557:	89 84 24 90 02 00 00 	mov    %eax,0x290(%rsp)
  40155e:	4c 89 bc 24 80 02 00 	mov    %r15,0x280(%rsp)
  401565:	00 
  401566:	44 8b 4c 24 10       	mov    0x10(%rsp),%r9d
  40156b:	44 8b 04 24          	mov    (%rsp),%r8d
  40156f:	48 8b b4 24 98 02 00 	mov    0x298(%rsp),%rsi
  401576:	00 
  401577:	41 be 01 00 00 00    	mov    $0x1,%r14d
  40157d:	45 33 ed             	xor    %r13d,%r13d
  401580:	41 83 f9 01          	cmp    $0x1,%r9d
  401584:	0f 82 55 0b 00 00    	jb     4020df <main+0xe4f>
  40158a:	45 89 d4             	mov    %r10d,%r12d
  40158d:	4c 89 d2             	mov    %r10,%rdx
  401590:	41 c1 e4 07          	shl    $0x7,%r12d
  401594:	44 89 e0             	mov    %r12d,%eax
  401597:	48 c1 e2 0a          	shl    $0xa,%rdx
  40159b:	f7 d8                	neg    %eax
  40159d:	05 88 13 00 00       	add    $0x1388,%eax
  4015a2:	3d 80 00 00 00       	cmp    $0x80,%eax
  4015a7:	44 89 a4 24 30 02 00 	mov    %r12d,0x230(%rsp)
  4015ae:	00 
  4015af:	44 89 ac 24 d0 01 00 	mov    %r13d,0x1d0(%rsp)
  4015b6:	00 
  4015b7:	48 8d 0c 13          	lea    (%rbx,%rdx,1),%rcx
  4015bb:	48 89 8c 24 a0 02 00 	mov    %rcx,0x2a0(%rsp)
  4015c2:	00 
  4015c3:	41 0f 43 c3          	cmovae %r11d,%eax
  4015c7:	48 03 d6             	add    %rsi,%rdx
  4015ca:	48 89 94 24 c8 01 00 	mov    %rdx,0x1c8(%rsp)
  4015d1:	00 
  4015d2:	4c 89 94 24 a8 01 00 	mov    %r10,0x1a8(%rsp)
  4015d9:	00 
  4015da:	48 89 b4 24 98 02 00 	mov    %rsi,0x298(%rsp)
  4015e1:	00 
  4015e2:	48 89 9c 24 00 02 00 	mov    %rbx,0x200(%rsp)
  4015e9:	00 
  4015ea:	be 01 00 00 00       	mov    $0x1,%esi
  4015ef:	45 33 db             	xor    %r11d,%r11d
  4015f2:	83 bc 24 b8 01 00 00 	cmpl   $0x1,0x1b8(%rsp)
  4015f9:	01 
  4015fa:	0f 82 5e 07 00 00    	jb     401d5e <main+0xace>
  401600:	44 8b ac 24 c0 01 00 	mov    0x1c0(%rsp),%r13d
  401607:	00 
  401608:	44 8b a4 24 d0 01 00 	mov    0x1d0(%rsp),%r12d
  40160f:	00 
  401610:	48 8b 9c 24 c8 01 00 	mov    0x1c8(%rsp),%rbx
  401617:	00 
  401618:	48 8b 8c 24 98 02 00 	mov    0x298(%rsp),%rcx
  40161f:	00 
  401620:	8b bc 24 30 02 00 00 	mov    0x230(%rsp),%edi
  401627:	43 8d 54 a5 00       	lea    0x0(%r13,%r12,4),%edx
  40162c:	48 63 d2             	movslq %edx,%rdx
  40162f:	4c 69 ea 40 9c 00 00 	imul   $0x9c40,%rdx,%r13
  401636:	44 8b 8c 24 20 02 00 	mov    0x220(%rsp),%r9d
  40163d:	00 
  40163e:	4a 8d 34 2b          	lea    (%rbx,%r13,1),%rsi
  401642:	48 89 b4 24 10 02 00 	mov    %rsi,0x210(%rsp)
  401649:	00 
  40164a:	4a 8d 9c 2b c0 d4 01 	lea    0x1d4c0(%rbx,%r13,1),%rbx
  401651:	00 
  401652:	48 83 e3 1f          	and    $0x1f,%rbx
  401656:	4e 8d 24 29          	lea    (%rcx,%r13,1),%r12
  40165a:	41 89 d8             	mov    %ebx,%r8d
  40165d:	89 c1                	mov    %eax,%ecx
  40165f:	45 89 c2             	mov    %r8d,%r10d
  401662:	45 89 c6             	mov    %r8d,%r14d
  401665:	41 f7 da             	neg    %r10d
  401668:	41 83 e6 07          	and    $0x7,%r14d
  40166c:	41 83 c2 20          	add    $0x20,%r10d
  401670:	41 c1 ea 03          	shr    $0x3,%r10d
  401674:	41 2b ca             	sub    %r10d,%ecx
  401677:	83 e1 03             	and    $0x3,%ecx
  40167a:	f7 d9                	neg    %ecx
  40167c:	03 c8                	add    %eax,%ecx
  40167e:	41 8d 52 04          	lea    0x4(%r10),%edx
  401682:	4c 03 ac 24 80 02 00 	add    0x280(%rsp),%r13
  401689:	00 
  40168a:	89 94 24 f0 01 00 00 	mov    %edx,0x1f0(%rsp)
  401691:	89 8c 24 b0 02 00 00 	mov    %ecx,0x2b0(%rsp)
  401698:	48 89 9c 24 f8 01 00 	mov    %rbx,0x1f8(%rsp)
  40169f:	00 
  4016a0:	43 8d 1c 99          	lea    (%r9,%r11,4),%ebx
  4016a4:	48 63 db             	movslq %ebx,%rbx
  4016a7:	c4 c1 7b 10 54 dd 08 	vmovsd 0x8(%r13,%rbx,8),%xmm2
  4016ae:	c5 fb 11 94 24 b8 02 	vmovsd %xmm2,0x2b8(%rsp)
  4016b5:	00 00 
  4016b7:	c4 41 7b 10 64 dd 00 	vmovsd 0x0(%r13,%rbx,8),%xmm12
  4016be:	c4 41 7b 10 4c dd 10 	vmovsd 0x10(%r13,%rbx,8),%xmm9
  4016c5:	c4 c1 7b 10 6c dd 18 	vmovsd 0x18(%r13,%rbx,8),%xmm5
  4016cc:	c4 41 7b 10 9c dd 40 	vmovsd 0x9c40(%r13,%rbx,8),%xmm11
  4016d3:	9c 00 00 
  4016d6:	c4 41 7b 10 b4 dd 48 	vmovsd 0x9c48(%r13,%rbx,8),%xmm14
  4016dd:	9c 00 00 
  4016e0:	c4 41 7b 10 84 dd 50 	vmovsd 0x9c50(%r13,%rbx,8),%xmm8
  4016e7:	9c 00 00 
  4016ea:	c4 c1 7b 10 a4 dd 58 	vmovsd 0x9c58(%r13,%rbx,8),%xmm4
  4016f1:	9c 00 00 
  4016f4:	c4 c1 7b 10 8c dd 80 	vmovsd 0x13880(%r13,%rbx,8),%xmm1
  4016fb:	38 01 00 
  4016fe:	c4 41 7b 10 ac dd 88 	vmovsd 0x13888(%r13,%rbx,8),%xmm13
  401705:	38 01 00 
  401708:	c4 c1 7b 10 bc dd 90 	vmovsd 0x13890(%r13,%rbx,8),%xmm7
  40170f:	38 01 00 
  401712:	c4 c1 7b 10 9c dd 98 	vmovsd 0x13898(%r13,%rbx,8),%xmm3
  401719:	38 01 00 
  40171c:	c4 c1 7b 10 84 dd c0 	vmovsd 0x1d4c0(%r13,%rbx,8),%xmm0
  401723:	d4 01 00 
  401726:	c4 41 7b 10 94 dd c8 	vmovsd 0x1d4c8(%r13,%rbx,8),%xmm10
  40172d:	d4 01 00 
  401730:	c4 c1 7b 10 b4 dd d0 	vmovsd 0x1d4d0(%r13,%rbx,8),%xmm6
  401737:	d4 01 00 
  40173a:	c4 c1 7b 10 94 dd d8 	vmovsd 0x1d4d8(%r13,%rbx,8),%xmm2
  401741:	d4 01 00 
  401744:	83 f8 04             	cmp    $0x4,%eax
  401747:	0f 82 50 0d 00 00    	jb     40249d <main+0x120d>
  40174d:	45 85 c0             	test   %r8d,%r8d
  401750:	0f 84 71 01 00 00    	je     4018c7 <main+0x637>
  401756:	45 85 f6             	test   %r14d,%r14d
  401759:	0f 85 3e 0d 00 00    	jne    40249d <main+0x120d>
  40175f:	3b 84 24 f0 01 00 00 	cmp    0x1f0(%rsp),%eax
  401766:	0f 82 31 0d 00 00    	jb     40249d <main+0x120d>
  40176c:	48 69 d3 40 9c 00 00 	imul   $0x9c40,%rbx,%rdx
  401773:	48 03 94 24 a0 02 00 	add    0x2a0(%rsp),%rdx
  40177a:	00 
  40177b:	45 33 ff             	xor    %r15d,%r15d
  40177e:	8b b4 24 b0 02 00 00 	mov    0x2b0(%rsp),%esi
  401785:	44 89 d1             	mov    %r10d,%ecx
  401788:	45 85 d2             	test   %r10d,%r10d
  40178b:	0f 84 4f 01 00 00    	je     4018e0 <main+0x650>
  401791:	c5 7b 11 94 24 78 03 	vmovsd %xmm10,0x378(%rsp)
  401798:	00 00 
  40179a:	c5 7b 11 ac 24 70 03 	vmovsd %xmm13,0x370(%rsp)
  4017a1:	00 00 
  4017a3:	c5 7b 11 b4 24 68 03 	vmovsd %xmm14,0x368(%rsp)
  4017aa:	00 00 
  4017ac:	c5 7b 11 a4 24 60 03 	vmovsd %xmm12,0x360(%rsp)
  4017b3:	00 00 
  4017b5:	4c 8b 8c 24 10 02 00 	mov    0x210(%rsp),%r9
  4017bc:	00 
  4017bd:	c4 21 7b 10 14 fa    	vmovsd (%rdx,%r15,8),%xmm10
  4017c3:	c4 01 7b 10 24 f9    	vmovsd (%r9,%r15,8),%xmm12
  4017c9:	c4 01 7b 10 ac f9 40 	vmovsd 0x9c40(%r9,%r15,8),%xmm13
  4017d0:	9c 00 00 
  4017d3:	c4 01 7b 10 b4 f9 80 	vmovsd 0x13880(%r9,%r15,8),%xmm14
  4017da:	38 01 00 
  4017dd:	c4 62 a9 b9 a4 24 60 	vfmadd231sd 0x360(%rsp),%xmm10,%xmm12
  4017e4:	03 00 00 
  4017e7:	c4 42 a1 b9 ea       	vfmadd231sd %xmm10,%xmm11,%xmm13
  4017ec:	c4 42 f1 b9 f2       	vfmadd231sd %xmm10,%xmm1,%xmm14
  4017f1:	c4 02 f9 a9 94 f9 c0 	vfmadd213sd 0x1d4c0(%r9,%r15,8),%xmm0,%xmm10
  4017f8:	d4 01 00 
  4017fb:	c4 21 7b 10 bc fa 40 	vmovsd 0x9c40(%rdx,%r15,8),%xmm15
  401802:	9c 00 00 
  401805:	c4 62 81 b9 a4 24 b8 	vfmadd231sd 0x2b8(%rsp),%xmm15,%xmm12
  40180c:	02 00 00 
  40180f:	c4 62 81 b9 ac 24 68 	vfmadd231sd 0x368(%rsp),%xmm15,%xmm13
  401816:	03 00 00 
  401819:	c4 62 81 b9 b4 24 70 	vfmadd231sd 0x370(%rsp),%xmm15,%xmm14
  401820:	03 00 00 
  401823:	c4 62 81 b9 94 24 78 	vfmadd231sd 0x378(%rsp),%xmm15,%xmm10
  40182a:	03 00 00 
  40182d:	c4 21 7b 10 bc fa 80 	vmovsd 0x13880(%rdx,%r15,8),%xmm15
  401834:	38 01 00 
  401837:	c4 42 81 b9 e1       	vfmadd231sd %xmm9,%xmm15,%xmm12
  40183c:	c4 42 b9 b9 ef       	vfmadd231sd %xmm15,%xmm8,%xmm13
  401841:	c4 42 c1 b9 f7       	vfmadd231sd %xmm15,%xmm7,%xmm14
  401846:	c4 42 c9 b9 d7       	vfmadd231sd %xmm15,%xmm6,%xmm10
  40184b:	c4 21 7b 10 bc fa c0 	vmovsd 0x1d4c0(%rdx,%r15,8),%xmm15
  401852:	d4 01 00 
  401855:	c4 62 81 b9 e5       	vfmadd231sd %xmm5,%xmm15,%xmm12
  40185a:	c4 42 d9 b9 ef       	vfmadd231sd %xmm15,%xmm4,%xmm13
  40185f:	c4 42 e1 b9 f7       	vfmadd231sd %xmm15,%xmm3,%xmm14
  401864:	c4 42 e9 b9 d7       	vfmadd231sd %xmm15,%xmm2,%xmm10
  401869:	c4 01 7b 11 24 f9    	vmovsd %xmm12,(%r9,%r15,8)
  40186f:	c4 01 7b 11 ac f9 40 	vmovsd %xmm13,0x9c40(%r9,%r15,8)
  401876:	9c 00 00 
  401879:	c4 01 7b 11 b4 f9 80 	vmovsd %xmm14,0x13880(%r9,%r15,8)
  401880:	38 01 00 
  401883:	c4 01 7b 11 94 f9 c0 	vmovsd %xmm10,0x1d4c0(%r9,%r15,8)
  40188a:	d4 01 00 
  40188d:	49 ff c7             	inc    %r15
  401890:	4c 3b f9             	cmp    %rcx,%r15
  401893:	0f 82 24 ff ff ff    	jb     4017bd <main+0x52d>
  401899:	c5 7b 10 94 24 78 03 	vmovsd 0x378(%rsp),%xmm10
  4018a0:	00 00 
  4018a2:	c5 7b 10 ac 24 70 03 	vmovsd 0x370(%rsp),%xmm13
  4018a9:	00 00 
  4018ab:	c5 7b 10 b4 24 68 03 	vmovsd 0x368(%rsp),%xmm14
  4018b2:	00 00 
  4018b4:	c5 7b 10 a4 24 60 03 	vmovsd 0x360(%rsp),%xmm12
  4018bb:	00 00 
  4018bd:	44 8b 8c 24 20 02 00 	mov    0x220(%rsp),%r9d
  4018c4:	00 
  4018c5:	eb 19                	jmp    4018e0 <main+0x650>
  4018c7:	48 69 d3 40 9c 00 00 	imul   $0x9c40,%rbx,%rdx
  4018ce:	48 03 94 24 a0 02 00 	add    0x2a0(%rsp),%rdx
  4018d5:	00 
  4018d6:	89 c6                	mov    %eax,%esi
  4018d8:	48 8b 8c 24 f8 01 00 	mov    0x1f8(%rsp),%rcx
  4018df:	00 
  4018e0:	c4 42 7d 19 fc       	vbroadcastsd %xmm12,%ymm15
  4018e5:	c5 fb 11 94 24 38 02 	vmovsd %xmm2,0x238(%rsp)
  4018ec:	00 00 
  4018ee:	c5 fb 11 9c 24 40 02 	vmovsd %xmm3,0x240(%rsp)
  4018f5:	00 00 
  4018f7:	c5 fb 11 a4 24 48 02 	vmovsd %xmm4,0x248(%rsp)
  4018fe:	00 00 
  401900:	c5 7d 11 7c 24 60    	vmovupd %ymm15,0x60(%rsp)
  401906:	c5 fb 11 ac 24 50 02 	vmovsd %xmm5,0x250(%rsp)
  40190d:	00 00 
  40190f:	c4 42 7d 19 fb       	vbroadcastsd %xmm11,%ymm15
  401914:	c5 fb 11 b4 24 58 02 	vmovsd %xmm6,0x258(%rsp)
  40191b:	00 00 
  40191d:	c5 fb 11 bc 24 60 02 	vmovsd %xmm7,0x260(%rsp)
  401924:	00 00 
  401926:	c5 7b 11 84 24 68 02 	vmovsd %xmm8,0x268(%rsp)
  40192d:	00 00 
  40192f:	c5 7d 11 bc 24 c0 02 	vmovupd %ymm15,0x2c0(%rsp)
  401936:	00 00 
  401938:	c5 7b 11 8c 24 70 02 	vmovsd %xmm9,0x270(%rsp)
  40193f:	00 00 
  401941:	c4 62 7d 19 f9       	vbroadcastsd %xmm1,%ymm15
  401946:	c5 7b 11 94 24 78 03 	vmovsd %xmm10,0x378(%rsp)
  40194d:	00 00 
  40194f:	c5 fb 11 84 24 80 03 	vmovsd %xmm0,0x380(%rsp)
  401956:	00 00 
  401958:	c5 fb 11 8c 24 88 03 	vmovsd %xmm1,0x388(%rsp)
  40195f:	00 00 
  401961:	c5 7d 11 bc 24 80 00 	vmovupd %ymm15,0x80(%rsp)
  401968:	00 00 
  40196a:	c5 7b 11 ac 24 70 03 	vmovsd %xmm13,0x370(%rsp)
  401971:	00 00 
  401973:	c4 62 7d 19 f8       	vbroadcastsd %xmm0,%ymm15
  401978:	c5 7b 11 b4 24 68 03 	vmovsd %xmm14,0x368(%rsp)
  40197f:	00 00 
  401981:	c5 7b 11 9c 24 78 02 	vmovsd %xmm11,0x278(%rsp)
  401988:	00 00 
  40198a:	c5 7b 11 a4 24 60 03 	vmovsd %xmm12,0x360(%rsp)
  401991:	00 00 
  401993:	c5 7d 11 bc 24 a0 00 	vmovupd %ymm15,0xa0(%rsp)
  40199a:	00 00 
  40199c:	c4 62 7d 19 bc 24 b8 	vbroadcastsd 0x2b8(%rsp),%ymm15
  4019a3:	02 00 00 
  4019a6:	c5 7d 11 bc 24 c0 00 	vmovupd %ymm15,0xc0(%rsp)
  4019ad:	00 00 
  4019af:	c4 42 7d 19 fe       	vbroadcastsd %xmm14,%ymm15
  4019b4:	c5 7d 11 bc 24 e0 02 	vmovupd %ymm15,0x2e0(%rsp)
  4019bb:	00 00 
  4019bd:	c4 42 7d 19 fd       	vbroadcastsd %xmm13,%ymm15
  4019c2:	c5 7d 11 bc 24 00 03 	vmovupd %ymm15,0x300(%rsp)
  4019c9:	00 00 
  4019cb:	c4 42 7d 19 fa       	vbroadcastsd %xmm10,%ymm15
  4019d0:	c5 7d 10 54 24 60    	vmovupd 0x60(%rsp),%ymm10
  4019d6:	c5 7d 11 bc 24 20 03 	vmovupd %ymm15,0x320(%rsp)
  4019dd:	00 00 
  4019df:	c4 42 7d 19 f9       	vbroadcastsd %xmm9,%ymm15
  4019e4:	c5 7d 10 8c 24 80 00 	vmovupd 0x80(%rsp),%ymm9
  4019eb:	00 00 
  4019ed:	c5 7d 11 bc 24 40 03 	vmovupd %ymm15,0x340(%rsp)
  4019f4:	00 00 
  4019f6:	c4 42 7d 19 f8       	vbroadcastsd %xmm8,%ymm15
  4019fb:	c5 7d 10 84 24 a0 00 	vmovupd 0xa0(%rsp),%ymm8
  401a02:	00 00 
  401a04:	c5 7d 11 bc 24 e0 00 	vmovupd %ymm15,0xe0(%rsp)
  401a0b:	00 00 
  401a0d:	c4 62 7d 19 ff       	vbroadcastsd %xmm7,%ymm15
  401a12:	c5 fd 10 bc 24 c0 00 	vmovupd 0xc0(%rsp),%ymm7
  401a19:	00 00 
  401a1b:	c5 7d 11 bc 24 00 01 	vmovupd %ymm15,0x100(%rsp)
  401a22:	00 00 
  401a24:	c4 62 7d 19 fe       	vbroadcastsd %xmm6,%ymm15
  401a29:	c5 fd 10 b4 24 e0 00 	vmovupd 0xe0(%rsp),%ymm6
  401a30:	00 00 
  401a32:	c5 7d 11 bc 24 20 01 	vmovupd %ymm15,0x120(%rsp)
  401a39:	00 00 
  401a3b:	c4 62 7d 19 fd       	vbroadcastsd %xmm5,%ymm15
  401a40:	c5 fd 10 ac 24 00 01 	vmovupd 0x100(%rsp),%ymm5
  401a47:	00 00 
  401a49:	c5 7d 11 bc 24 40 01 	vmovupd %ymm15,0x140(%rsp)
  401a50:	00 00 
  401a52:	c4 62 7d 19 fc       	vbroadcastsd %xmm4,%ymm15
  401a57:	c5 fd 10 a4 24 20 01 	vmovupd 0x120(%rsp),%ymm4
  401a5e:	00 00 
  401a60:	c5 7d 11 bc 24 60 01 	vmovupd %ymm15,0x160(%rsp)
  401a67:	00 00 
  401a69:	c4 62 7d 19 fb       	vbroadcastsd %xmm3,%ymm15
  401a6e:	c5 fd 10 9c 24 40 01 	vmovupd 0x140(%rsp),%ymm3
  401a75:	00 00 
  401a77:	c5 7d 11 bc 24 80 01 	vmovupd %ymm15,0x180(%rsp)
  401a7e:	00 00 
  401a80:	c4 62 7d 19 fa       	vbroadcastsd %xmm2,%ymm15
  401a85:	c5 fd 10 94 24 60 01 	vmovupd 0x160(%rsp),%ymm2
  401a8c:	00 00 
  401a8e:	c5 fd 10 8c 24 80 01 	vmovupd 0x180(%rsp),%ymm1
  401a95:	00 00 
  401a97:	4c 63 fe             	movslq %esi,%r15
  401a9a:	c4 c1 7d 6f c7       	vmovdqa %ymm15,%ymm0
  401a9f:	4c 8b 8c 24 10 02 00 	mov    0x210(%rsp),%r9
  401aa6:	00 
  401aa7:	c5 7d 10 3c ca       	vmovupd (%rdx,%rcx,8),%ymm15
  401aac:	c4 41 7d 10 ac c9 40 	vmovupd 0x9c40(%r9,%rcx,8),%ymm13
  401ab3:	9c 00 00 
  401ab6:	c4 41 7d 10 24 c9    	vmovupd (%r9,%rcx,8),%ymm12
  401abc:	c4 41 7d 10 b4 c9 80 	vmovupd 0x13880(%r9,%rcx,8),%ymm14
  401ac3:	38 01 00 
  401ac6:	c4 41 7d 10 9c c9 c0 	vmovupd 0x1d4c0(%r9,%rcx,8),%ymm11
  401acd:	d4 01 00 
  401ad0:	c4 62 85 b8 ac 24 c0 	vfmadd231pd 0x2c0(%rsp),%ymm15,%ymm13
  401ad7:	02 00 00 
  401ada:	c4 42 85 b8 e2       	vfmadd231pd %ymm10,%ymm15,%ymm12
  401adf:	c4 42 85 b8 f1       	vfmadd231pd %ymm9,%ymm15,%ymm14
  401ae4:	c4 42 85 b8 d8       	vfmadd231pd %ymm8,%ymm15,%ymm11
  401ae9:	c5 7d 10 bc ca 40 9c 	vmovupd 0x9c40(%rdx,%rcx,8),%ymm15
  401af0:	00 00 
  401af2:	c4 62 85 b8 ac 24 e0 	vfmadd231pd 0x2e0(%rsp),%ymm15,%ymm13
  401af9:	02 00 00 
  401afc:	c4 62 85 b8 b4 24 00 	vfmadd231pd 0x300(%rsp),%ymm15,%ymm14
  401b03:	03 00 00 
  401b06:	c4 62 85 b8 9c 24 20 	vfmadd231pd 0x320(%rsp),%ymm15,%ymm11
  401b0d:	03 00 00 
  401b10:	c4 62 85 b8 e7       	vfmadd231pd %ymm7,%ymm15,%ymm12
  401b15:	c5 7d 10 bc ca 80 38 	vmovupd 0x13880(%rdx,%rcx,8),%ymm15
  401b1c:	01 00 
  401b1e:	c4 62 85 b8 a4 24 40 	vfmadd231pd 0x340(%rsp),%ymm15,%ymm12
  401b25:	03 00 00 
  401b28:	c4 62 85 b8 ee       	vfmadd231pd %ymm6,%ymm15,%ymm13
  401b2d:	c4 62 85 b8 f5       	vfmadd231pd %ymm5,%ymm15,%ymm14
  401b32:	c4 62 85 b8 dc       	vfmadd231pd %ymm4,%ymm15,%ymm11
  401b37:	c5 7d 10 bc ca c0 d4 	vmovupd 0x1d4c0(%rdx,%rcx,8),%ymm15
  401b3e:	01 00 
  401b40:	c4 62 85 b8 e3       	vfmadd231pd %ymm3,%ymm15,%ymm12
  401b45:	c4 62 85 b8 ea       	vfmadd231pd %ymm2,%ymm15,%ymm13
  401b4a:	c4 62 85 b8 f1       	vfmadd231pd %ymm1,%ymm15,%ymm14
  401b4f:	c4 62 85 b8 d8       	vfmadd231pd %ymm0,%ymm15,%ymm11
  401b54:	c4 41 7d 11 24 c9    	vmovupd %ymm12,(%r9,%rcx,8)
  401b5a:	c4 41 7d 11 ac c9 40 	vmovupd %ymm13,0x9c40(%r9,%rcx,8)
  401b61:	9c 00 00 
  401b64:	c4 41 7d 11 b4 c9 80 	vmovupd %ymm14,0x13880(%r9,%rcx,8)
  401b6b:	38 01 00 
  401b6e:	c4 41 7d 11 9c c9 c0 	vmovupd %ymm11,0x1d4c0(%r9,%rcx,8)
  401b75:	d4 01 00 
  401b78:	48 83 c1 04          	add    $0x4,%rcx
  401b7c:	49 3b cf             	cmp    %r15,%rcx
  401b7f:	0f 82 22 ff ff ff    	jb     401aa7 <main+0x817>
  401b85:	c5 fb 10 94 24 38 02 	vmovsd 0x238(%rsp),%xmm2
  401b8c:	00 00 
  401b8e:	c5 fb 10 9c 24 40 02 	vmovsd 0x240(%rsp),%xmm3
  401b95:	00 00 
  401b97:	c5 fb 10 a4 24 48 02 	vmovsd 0x248(%rsp),%xmm4
  401b9e:	00 00 
  401ba0:	c5 fb 10 ac 24 50 02 	vmovsd 0x250(%rsp),%xmm5
  401ba7:	00 00 
  401ba9:	c5 fb 10 b4 24 58 02 	vmovsd 0x258(%rsp),%xmm6
  401bb0:	00 00 
  401bb2:	c5 fb 10 bc 24 60 02 	vmovsd 0x260(%rsp),%xmm7
  401bb9:	00 00 
  401bbb:	c5 7b 10 84 24 68 02 	vmovsd 0x268(%rsp),%xmm8
  401bc2:	00 00 
  401bc4:	c5 7b 10 8c 24 70 02 	vmovsd 0x270(%rsp),%xmm9
  401bcb:	00 00 
  401bcd:	c5 7b 10 94 24 78 03 	vmovsd 0x378(%rsp),%xmm10
  401bd4:	00 00 
  401bd6:	c5 7b 10 ac 24 70 03 	vmovsd 0x370(%rsp),%xmm13
  401bdd:	00 00 
  401bdf:	c5 7b 10 b4 24 68 03 	vmovsd 0x368(%rsp),%xmm14
  401be6:	00 00 
  401be8:	c5 fb 10 84 24 80 03 	vmovsd 0x380(%rsp),%xmm0
  401bef:	00 00 
  401bf1:	c5 fb 10 8c 24 88 03 	vmovsd 0x388(%rsp),%xmm1
  401bf8:	00 00 
  401bfa:	c5 7b 10 9c 24 78 02 	vmovsd 0x278(%rsp),%xmm11
  401c01:	00 00 
  401c03:	c5 7b 10 a4 24 60 03 	vmovsd 0x360(%rsp),%xmm12
  401c0a:	00 00 
  401c0c:	44 8b 8c 24 20 02 00 	mov    0x220(%rsp),%r9d
  401c13:	00 
  401c14:	33 c9                	xor    %ecx,%ecx
  401c16:	8d 56 01             	lea    0x1(%rsi),%edx
  401c19:	3b d0                	cmp    %eax,%edx
  401c1b:	0f 87 1d 01 00 00    	ja     401d3e <main+0xaae>
  401c21:	48 69 d3 40 9c 00 00 	imul   $0x9c40,%rbx,%rdx
  401c28:	48 03 94 24 00 02 00 	add    0x200(%rsp),%rdx
  401c2f:	00 
  401c30:	8d 1c 3e             	lea    (%rsi,%rdi,1),%ebx
  401c33:	48 63 db             	movslq %ebx,%rbx
  401c36:	f7 de                	neg    %esi
  401c38:	03 f0                	add    %eax,%esi
  401c3a:	48 63 f6             	movslq %esi,%rsi
  401c3d:	c5 7b 11 ac 24 70 03 	vmovsd %xmm13,0x370(%rsp)
  401c44:	00 00 
  401c46:	c5 7b 11 b4 24 68 03 	vmovsd %xmm14,0x368(%rsp)
  401c4d:	00 00 
  401c4f:	48 8d 14 da          	lea    (%rdx,%rbx,8),%rdx
  401c53:	c5 fb 11 84 24 80 03 	vmovsd %xmm0,0x380(%rsp)
  401c5a:	00 00 
  401c5c:	49 8d 1c dc          	lea    (%r12,%rbx,8),%rbx
  401c60:	c5 fb 11 8c 24 88 03 	vmovsd %xmm1,0x388(%rsp)
  401c67:	00 00 
  401c69:	c5 7b 10 3c ca       	vmovsd (%rdx,%rcx,8),%xmm15
  401c6e:	c5 fb 10 0c cb       	vmovsd (%rbx,%rcx,8),%xmm1
  401c73:	c5 7b 10 ac cb 40 9c 	vmovsd 0x9c40(%rbx,%rcx,8),%xmm13
  401c7a:	00 00 
  401c7c:	c5 7b 10 b4 cb 80 38 	vmovsd 0x13880(%rbx,%rcx,8),%xmm14
  401c83:	01 00 
  401c85:	c5 fb 10 84 cb c0 d4 	vmovsd 0x1d4c0(%rbx,%rcx,8),%xmm0
  401c8c:	01 00 
  401c8e:	c4 c2 81 b9 cc       	vfmadd231sd %xmm12,%xmm15,%xmm1
  401c93:	c4 42 81 b9 eb       	vfmadd231sd %xmm11,%xmm15,%xmm13
  401c98:	c4 62 81 b9 b4 24 88 	vfmadd231sd 0x388(%rsp),%xmm15,%xmm14
  401c9f:	03 00 00 
  401ca2:	c4 e2 81 b9 84 24 80 	vfmadd231sd 0x380(%rsp),%xmm15,%xmm0
  401ca9:	03 00 00 
  401cac:	c5 7b 10 bc ca 40 9c 	vmovsd 0x9c40(%rdx,%rcx,8),%xmm15
  401cb3:	00 00 
  401cb5:	c4 e2 81 b9 8c 24 b8 	vfmadd231sd 0x2b8(%rsp),%xmm15,%xmm1
  401cbc:	02 00 00 
  401cbf:	c4 62 81 b9 ac 24 68 	vfmadd231sd 0x368(%rsp),%xmm15,%xmm13
  401cc6:	03 00 00 
  401cc9:	c4 62 81 b9 b4 24 70 	vfmadd231sd 0x370(%rsp),%xmm15,%xmm14
  401cd0:	03 00 00 
  401cd3:	c4 c2 81 b9 c2       	vfmadd231sd %xmm10,%xmm15,%xmm0
  401cd8:	c5 7b 10 bc ca 80 38 	vmovsd 0x13880(%rdx,%rcx,8),%xmm15
  401cdf:	01 00 
  401ce1:	c4 c2 81 b9 c9       	vfmadd231sd %xmm9,%xmm15,%xmm1
  401ce6:	c4 42 81 b9 e8       	vfmadd231sd %xmm8,%xmm15,%xmm13
  401ceb:	c4 62 81 b9 f7       	vfmadd231sd %xmm7,%xmm15,%xmm14
  401cf0:	c4 e2 81 b9 c6       	vfmadd231sd %xmm6,%xmm15,%xmm0
  401cf5:	c5 7b 10 bc ca c0 d4 	vmovsd 0x1d4c0(%rdx,%rcx,8),%xmm15
  401cfc:	01 00 
  401cfe:	c4 e2 81 b9 cd       	vfmadd231sd %xmm5,%xmm15,%xmm1
  401d03:	c4 62 81 b9 ec       	vfmadd231sd %xmm4,%xmm15,%xmm13
  401d08:	c4 62 81 b9 f3       	vfmadd231sd %xmm3,%xmm15,%xmm14
  401d0d:	c4 e2 81 b9 c2       	vfmadd231sd %xmm2,%xmm15,%xmm0
  401d12:	c5 fb 11 0c cb       	vmovsd %xmm1,(%rbx,%rcx,8)
  401d17:	c5 7b 11 ac cb 40 9c 	vmovsd %xmm13,0x9c40(%rbx,%rcx,8)
  401d1e:	00 00 
  401d20:	c5 7b 11 b4 cb 80 38 	vmovsd %xmm14,0x13880(%rbx,%rcx,8)
  401d27:	01 00 
  401d29:	c5 fb 11 84 cb c0 d4 	vmovsd %xmm0,0x1d4c0(%rbx,%rcx,8)
  401d30:	01 00 
  401d32:	48 ff c1             	inc    %rcx
  401d35:	48 3b ce             	cmp    %rsi,%rcx
  401d38:	0f 82 2b ff ff ff    	jb     401c69 <main+0x9d9>
  401d3e:	41 ff c3             	inc    %r11d
  401d41:	44 3b 9c 24 b8 01 00 	cmp    0x1b8(%rsp),%r11d
  401d48:	00 
  401d49:	0f 82 51 f9 ff ff    	jb     4016a0 <main+0x410>
  401d4f:	89 bc 24 30 02 00 00 	mov    %edi,0x230(%rsp)
  401d56:	42 8d 34 9d 01 00 00 	lea    0x1(,%r11,4),%esi
  401d5d:	00 
  401d5e:	45 33 c9             	xor    %r9d,%r9d
  401d61:	3b b4 24 b0 01 00 00 	cmp    0x1b0(%rsp),%esi
  401d68:	0f 87 25 03 00 00    	ja     402093 <main+0xe03>
  401d6e:	8b 8c 24 c0 01 00 00 	mov    0x1c0(%rsp),%ecx
  401d75:	41 89 c7             	mov    %eax,%r15d
  401d78:	44 8b a4 24 d0 01 00 	mov    0x1d0(%rsp),%r12d
  401d7f:	00 
  401d80:	44 8b 9c 24 20 02 00 	mov    0x220(%rsp),%r11d
  401d87:	00 
  401d88:	4c 8b 84 24 c8 01 00 	mov    0x1c8(%rsp),%r8
  401d8f:	00 
  401d90:	48 8b bc 24 98 02 00 	mov    0x298(%rsp),%rdi
  401d97:	00 
  401d98:	42 8d 14 a1          	lea    (%rcx,%r12,4),%edx
  401d9c:	48 63 d2             	movslq %edx,%rdx
  401d9f:	41 8d 1c 33          	lea    (%r11,%rsi,1),%ebx
  401da3:	4c 69 ea 40 9c 00 00 	imul   $0x9c40,%rdx,%r13
  401daa:	48 63 db             	movslq %ebx,%rbx
  401dad:	4b 8d 94 28 c0 d4 01 	lea    0x1d4c0(%r8,%r13,1),%rdx
  401db4:	00 
  401db5:	4c 69 db 40 9c 00 00 	imul   $0x9c40,%rbx,%r11
  401dbc:	48 83 e2 1f          	and    $0x1f,%rdx
  401dc0:	4e 8d 24 2f          	lea    (%rdi,%r13,1),%r12
  401dc4:	89 d7                	mov    %edx,%edi
  401dc6:	4b 8d 0c 28          	lea    (%r8,%r13,1),%rcx
  401dca:	41 89 f8             	mov    %edi,%r8d
  401dcd:	f7 de                	neg    %esi
  401dcf:	41 f7 d8             	neg    %r8d
  401dd2:	41 89 fe             	mov    %edi,%r14d
  401dd5:	41 83 c0 20          	add    $0x20,%r8d
  401dd9:	41 83 e6 07          	and    $0x7,%r14d
  401ddd:	41 c1 e8 03          	shr    $0x3,%r8d
  401de1:	45 2b f8             	sub    %r8d,%r15d
  401de4:	4c 03 ac 24 80 02 00 	add    0x280(%rsp),%r13
  401deb:	00 
  401dec:	41 83 e7 03          	and    $0x3,%r15d
  401df0:	03 b4 24 b0 01 00 00 	add    0x1b0(%rsp),%esi
  401df7:	41 f7 df             	neg    %r15d
  401dfa:	4c 8b 94 24 00 02 00 	mov    0x200(%rsp),%r10
  401e01:	00 
  401e02:	44 03 f8             	add    %eax,%r15d
  401e05:	48 63 f6             	movslq %esi,%rsi
  401e08:	4d 8d 6c dd 00       	lea    0x0(%r13,%rbx,8),%r13
  401e0d:	48 ff c6             	inc    %rsi
  401e10:	41 8d 58 04          	lea    0x4(%r8),%ebx
  401e14:	48 89 b4 24 e0 01 00 	mov    %rsi,0x1e0(%rsp)
  401e1b:	00 
  401e1c:	4d 03 d3             	add    %r11,%r10
  401e1f:	4c 03 9c 24 a0 02 00 	add    0x2a0(%rsp),%r11
  401e26:	00 
  401e27:	8b b4 24 30 02 00 00 	mov    0x230(%rsp),%esi
  401e2e:	89 9c 24 d8 01 00 00 	mov    %ebx,0x1d8(%rsp)
  401e35:	44 89 bc 24 a8 02 00 	mov    %r15d,0x2a8(%rsp)
  401e3c:	00 
  401e3d:	48 89 94 24 e8 01 00 	mov    %rdx,0x1e8(%rsp)
  401e44:	00 
  401e45:	48 89 8c 24 08 02 00 	mov    %rcx,0x208(%rsp)
  401e4c:	00 
  401e4d:	c4 81 7b 10 4c cd f8 	vmovsd -0x8(%r13,%r9,8),%xmm1
  401e54:	c4 81 7b 10 9c cd 38 	vmovsd 0x9c38(%r13,%r9,8),%xmm3
  401e5b:	9c 00 00 
  401e5e:	c4 81 7b 10 84 cd 78 	vmovsd 0x13878(%r13,%r9,8),%xmm0
  401e65:	38 01 00 
  401e68:	c4 81 7b 10 94 cd b8 	vmovsd 0x1d4b8(%r13,%r9,8),%xmm2
  401e6f:	d4 01 00 
  401e72:	83 f8 04             	cmp    $0x4,%eax
  401e75:	0f 82 29 06 00 00    	jb     4024a4 <main+0x1214>
  401e7b:	85 ff                	test   %edi,%edi
  401e7d:	0f 84 ae 00 00 00    	je     401f31 <main+0xca1>
  401e83:	45 85 f6             	test   %r14d,%r14d
  401e86:	0f 85 18 06 00 00    	jne    4024a4 <main+0x1214>
  401e8c:	3b 84 24 d8 01 00 00 	cmp    0x1d8(%rsp),%eax
  401e93:	0f 82 0b 06 00 00    	jb     4024a4 <main+0x1214>
  401e99:	8b 9c 24 a8 02 00 00 	mov    0x2a8(%rsp),%ebx
  401ea0:	33 c9                	xor    %ecx,%ecx
  401ea2:	45 89 c7             	mov    %r8d,%r15d
  401ea5:	45 85 c0             	test   %r8d,%r8d
  401ea8:	0f 84 8d 00 00 00    	je     401f3b <main+0xcab>
  401eae:	89 b4 24 30 02 00 00 	mov    %esi,0x230(%rsp)
  401eb5:	4c 89 da             	mov    %r11,%rdx
  401eb8:	48 8b b4 24 08 02 00 	mov    0x208(%rsp),%rsi
  401ebf:	00 
  401ec0:	c5 7b 10 82 c0 63 ff 	vmovsd -0x9c40(%rdx),%xmm8
  401ec7:	ff 
  401ec8:	48 83 c2 08          	add    $0x8,%rdx
  401ecc:	c5 fb 10 24 ce       	vmovsd (%rsi,%rcx,8),%xmm4
  401ed1:	c5 fb 10 ac ce 40 9c 	vmovsd 0x9c40(%rsi,%rcx,8),%xmm5
  401ed8:	00 00 
  401eda:	c5 fb 10 b4 ce 80 38 	vmovsd 0x13880(%rsi,%rcx,8),%xmm6
  401ee1:	01 00 
  401ee3:	c5 fb 10 bc ce c0 d4 	vmovsd 0x1d4c0(%rsi,%rcx,8),%xmm7
  401eea:	01 00 
  401eec:	c4 e2 b9 b9 e1       	vfmadd231sd %xmm1,%xmm8,%xmm4
  401ef1:	c4 c2 e1 b9 e8       	vfmadd231sd %xmm8,%xmm3,%xmm5
  401ef6:	c4 c2 f9 b9 f0       	vfmadd231sd %xmm8,%xmm0,%xmm6
  401efb:	c4 62 e9 a9 c7       	vfmadd213sd %xmm7,%xmm2,%xmm8
  401f00:	c5 fb 11 24 ce       	vmovsd %xmm4,(%rsi,%rcx,8)
  401f05:	c5 fb 11 ac ce 40 9c 	vmovsd %xmm5,0x9c40(%rsi,%rcx,8)
  401f0c:	00 00 
  401f0e:	c5 fb 11 b4 ce 80 38 	vmovsd %xmm6,0x13880(%rsi,%rcx,8)
  401f15:	01 00 
  401f17:	c5 7b 11 84 ce c0 d4 	vmovsd %xmm8,0x1d4c0(%rsi,%rcx,8)
  401f1e:	01 00 
  401f20:	48 ff c1             	inc    %rcx
  401f23:	49 3b cf             	cmp    %r15,%rcx
  401f26:	72 98                	jb     401ec0 <main+0xc30>
  401f28:	8b b4 24 30 02 00 00 	mov    0x230(%rsp),%esi
  401f2f:	eb 0a                	jmp    401f3b <main+0xcab>
  401f31:	4c 8b bc 24 e8 01 00 	mov    0x1e8(%rsp),%r15
  401f38:	00 
  401f39:	89 c3                	mov    %eax,%ebx
  401f3b:	c4 e2 7d 19 f9       	vbroadcastsd %xmm1,%ymm7
  401f40:	c4 e2 7d 19 f3       	vbroadcastsd %xmm3,%ymm6
  401f45:	c4 e2 7d 19 e8       	vbroadcastsd %xmm0,%ymm5
  401f4a:	c4 e2 7d 19 e2       	vbroadcastsd %xmm2,%ymm4
  401f4f:	48 63 d3             	movslq %ebx,%rdx
  401f52:	48 8b 8c 24 08 02 00 	mov    0x208(%rsp),%rcx
  401f59:	00 
  401f5a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401f60:	c4 21 7d 10 04 f9    	vmovupd (%rcx,%r15,8),%ymm8
  401f66:	c4 01 7d 10 9c fb c0 	vmovupd -0x9c40(%r11,%r15,8),%ymm11
  401f6d:	63 ff ff 
  401f70:	c4 21 7d 10 8c f9 40 	vmovupd 0x9c40(%rcx,%r15,8),%ymm9
  401f77:	9c 00 00 
  401f7a:	c4 21 7d 10 94 f9 80 	vmovupd 0x13880(%rcx,%r15,8),%ymm10
  401f81:	38 01 00 
  401f84:	c4 21 7d 10 a4 f9 c0 	vmovupd 0x1d4c0(%rcx,%r15,8),%ymm12
  401f8b:	d4 01 00 
  401f8e:	c4 62 a5 b8 c7       	vfmadd231pd %ymm7,%ymm11,%ymm8
  401f93:	c4 62 a5 b8 ce       	vfmadd231pd %ymm6,%ymm11,%ymm9
  401f98:	c4 62 a5 b8 d5       	vfmadd231pd %ymm5,%ymm11,%ymm10
  401f9d:	c4 62 a5 b8 e4       	vfmadd231pd %ymm4,%ymm11,%ymm12
  401fa2:	c4 21 7d 11 04 f9    	vmovupd %ymm8,(%rcx,%r15,8)
  401fa8:	c4 21 7d 11 8c f9 40 	vmovupd %ymm9,0x9c40(%rcx,%r15,8)
  401faf:	9c 00 00 
  401fb2:	c4 21 7d 11 94 f9 80 	vmovupd %ymm10,0x13880(%rcx,%r15,8)
  401fb9:	38 01 00 
  401fbc:	c4 21 7d 11 a4 f9 c0 	vmovupd %ymm12,0x1d4c0(%rcx,%r15,8)
  401fc3:	d4 01 00 
  401fc6:	49 83 c7 04          	add    $0x4,%r15
  401fca:	4c 3b fa             	cmp    %rdx,%r15
  401fcd:	72 91                	jb     401f60 <main+0xcd0>
  401fcf:	33 d2                	xor    %edx,%edx
  401fd1:	8d 4b 01             	lea    0x1(%rbx),%ecx
  401fd4:	3b c8                	cmp    %eax,%ecx
  401fd6:	0f 87 91 00 00 00    	ja     40206d <main+0xddd>
  401fdc:	8d 0c 33             	lea    (%rbx,%rsi,1),%ecx
  401fdf:	f7 db                	neg    %ebx
  401fe1:	48 63 c9             	movslq %ecx,%rcx
  401fe4:	03 d8                	add    %eax,%ebx
  401fe6:	48 63 db             	movslq %ebx,%rbx
  401fe9:	4d 8d 3c cc          	lea    (%r12,%rcx,8),%r15
  401fed:	49 8d 0c ca          	lea    (%r10,%rcx,8),%rcx
  401ff1:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401ff8:	00 
  401ff9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  402000:	c5 fb 10 bc d1 c0 63 	vmovsd -0x9c40(%rcx,%rdx,8),%xmm7
  402007:	ff ff 
  402009:	c4 c1 7b 10 24 d7    	vmovsd (%r15,%rdx,8),%xmm4
  40200f:	c4 c1 7b 10 ac d7 40 	vmovsd 0x9c40(%r15,%rdx,8),%xmm5
  402016:	9c 00 00 
  402019:	c4 c1 7b 10 b4 d7 80 	vmovsd 0x13880(%r15,%rdx,8),%xmm6
  402020:	38 01 00 
  402023:	c4 41 7b 10 84 d7 c0 	vmovsd 0x1d4c0(%r15,%rdx,8),%xmm8
  40202a:	d4 01 00 
  40202d:	c4 e2 c1 b9 e1       	vfmadd231sd %xmm1,%xmm7,%xmm4
  402032:	c4 e2 c1 b9 eb       	vfmadd231sd %xmm3,%xmm7,%xmm5
  402037:	c4 e2 c1 b9 f0       	vfmadd231sd %xmm0,%xmm7,%xmm6
  40203c:	c4 62 c1 b9 c2       	vfmadd231sd %xmm2,%xmm7,%xmm8
  402041:	c4 c1 7b 11 24 d7    	vmovsd %xmm4,(%r15,%rdx,8)
  402047:	c4 c1 7b 11 ac d7 40 	vmovsd %xmm5,0x9c40(%r15,%rdx,8)
  40204e:	9c 00 00 
  402051:	c4 c1 7b 11 b4 d7 80 	vmovsd %xmm6,0x13880(%r15,%rdx,8)
  402058:	38 01 00 
  40205b:	c4 41 7b 11 84 d7 c0 	vmovsd %xmm8,0x1d4c0(%r15,%rdx,8)
  402062:	d4 01 00 
  402065:	48 ff c2             	inc    %rdx
  402068:	48 3b d3             	cmp    %rbx,%rdx
  40206b:	72 93                	jb     402000 <main+0xd70>
  40206d:	49 ff c1             	inc    %r9
  402070:	49 81 c3 40 9c 00 00 	add    $0x9c40,%r11
  402077:	49 81 c2 40 9c 00 00 	add    $0x9c40,%r10
  40207e:	4c 3b 8c 24 e0 01 00 	cmp    0x1e0(%rsp),%r9
  402085:	00 
  402086:	0f 82 c1 fd ff ff    	jb     401e4d <main+0xbbd>
  40208c:	89 b4 24 30 02 00 00 	mov    %esi,0x230(%rsp)
  402093:	8b 94 24 d0 01 00 00 	mov    0x1d0(%rsp),%edx
  40209a:	ff c2                	inc    %edx
  40209c:	89 94 24 d0 01 00 00 	mov    %edx,0x1d0(%rsp)
  4020a3:	3b 54 24 10          	cmp    0x10(%rsp),%edx
  4020a7:	0f 82 3d f5 ff ff    	jb     4015ea <main+0x35a>
  4020ad:	41 89 d5             	mov    %edx,%r13d
  4020b0:	41 bb 80 00 00 00    	mov    $0x80,%r11d
  4020b6:	4c 8b 94 24 a8 01 00 	mov    0x1a8(%rsp),%r10
  4020bd:	00 
  4020be:	46 8d 34 ad 01 00 00 	lea    0x1(,%r13,4),%r14d
  4020c5:	00 
  4020c6:	44 8b 4c 24 10       	mov    0x10(%rsp),%r9d
  4020cb:	44 8b 04 24          	mov    (%rsp),%r8d
  4020cf:	48 8b b4 24 98 02 00 	mov    0x298(%rsp),%rsi
  4020d6:	00 
  4020d7:	48 8b 9c 24 00 02 00 	mov    0x200(%rsp),%rbx
  4020de:	00 
  4020df:	45 33 ff             	xor    %r15d,%r15d
  4020e2:	33 d2                	xor    %edx,%edx
  4020e4:	45 3b f0             	cmp    %r8d,%r14d
  4020e7:	0f 87 38 02 00 00    	ja     402325 <main+0x1095>
  4020ed:	8b 84 24 c0 01 00 00 	mov    0x1c0(%rsp),%eax
  4020f4:	4d 89 d5             	mov    %r10,%r13
  4020f7:	49 c1 e5 0a          	shl    $0xa,%r13
  4020fb:	4c 8b 64 24 20       	mov    0x20(%rsp),%r12
  402100:	4c 89 7c 24 30       	mov    %r15,0x30(%rsp)
  402105:	42 8d 0c 30          	lea    (%rax,%r14,1),%ecx
  402109:	48 63 c9             	movslq %ecx,%rcx
  40210c:	41 f7 de             	neg    %r14d
  40210f:	48 69 c1 40 9c 00 00 	imul   $0x9c40,%rcx,%rax
  402116:	44 89 d1             	mov    %r10d,%ecx
  402119:	45 03 f0             	add    %r8d,%r14d
  40211c:	c1 e1 07             	shl    $0x7,%ecx
  40211f:	4a 8d 3c 2e          	lea    (%rsi,%r13,1),%rdi
  402123:	f7 d9                	neg    %ecx
  402125:	4c 03 e0             	add    %rax,%r12
  402128:	81 c1 88 13 00 00    	add    $0x1388,%ecx
  40212e:	48 03 c7             	add    %rdi,%rax
  402131:	81 f9 80 00 00 00    	cmp    $0x80,%ecx
  402137:	4d 63 f6             	movslq %r14d,%r14
  40213a:	41 0f 43 cb          	cmovae %r11d,%ecx
  40213e:	49 ff c6             	inc    %r14
  402141:	4c 03 6c 24 18       	add    0x18(%rsp),%r13
  402146:	48 63 f9             	movslq %ecx,%rdi
  402149:	4c 89 6c 24 40       	mov    %r13,0x40(%rsp)
  40214e:	4c 89 74 24 28       	mov    %r14,0x28(%rsp)
  402153:	48 89 7c 24 38       	mov    %rdi,0x38(%rsp)
  402158:	4c 89 64 24 48       	mov    %r12,0x48(%rsp)
  40215d:	4c 89 94 24 a8 01 00 	mov    %r10,0x1a8(%rsp)
  402164:	00 
  402165:	48 89 b4 24 98 02 00 	mov    %rsi,0x298(%rsp)
  40216c:	00 
  40216d:	48 89 9c 24 00 02 00 	mov    %rbx,0x200(%rsp)
  402174:	00 
  402175:	33 db                	xor    %ebx,%ebx
  402177:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
  40217c:	4c 8d b4 10 c0 63 ff 	lea    -0x9c40(%rax,%rdx,1),%r14
  402183:	ff 
  402184:	49 83 e6 1f          	and    $0x1f,%r14
  402188:	4c 8d 3c 10          	lea    (%rax,%rdx,1),%r15
  40218c:	45 89 f2             	mov    %r14d,%r10d
  40218f:	49 89 d8             	mov    %rbx,%r8
  402192:	45 89 d4             	mov    %r10d,%r12d
  402195:	4c 8d 0c 16          	lea    (%rsi,%rdx,1),%r9
  402199:	41 f7 dc             	neg    %r12d
  40219c:	89 ce                	mov    %ecx,%esi
  40219e:	41 83 c4 20          	add    $0x20,%r12d
  4021a2:	45 89 d3             	mov    %r10d,%r11d
  4021a5:	41 c1 ec 03          	shr    $0x3,%r12d
  4021a9:	41 83 e3 07          	and    $0x7,%r11d
  4021ad:	41 2b f4             	sub    %r12d,%esi
  4021b0:	83 e6 03             	and    $0x3,%esi
  4021b3:	f7 de                	neg    %esi
  4021b5:	48 89 54 24 58       	mov    %rdx,0x58(%rsp)
  4021ba:	45 8d 6c 24 04       	lea    0x4(%r12),%r13d
  4021bf:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  4021c4:	03 f1                	add    %ecx,%esi
  4021c6:	48 8b 5c 24 38       	mov    0x38(%rsp),%rbx
  4021cb:	48 8b 94 24 28 02 00 	mov    0x228(%rsp),%rdx
  4021d2:	00 
  4021d3:	4c 89 b4 24 a0 01 00 	mov    %r14,0x1a0(%rsp)
  4021da:	00 
  4021db:	4c 89 bc 24 18 02 00 	mov    %r15,0x218(%rsp)
  4021e2:	00 
  4021e3:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
  4021e8:	c4 81 7b 10 8c c1 c0 	vmovsd -0x9c40(%r9,%r8,8),%xmm1
  4021ef:	63 ff ff 
  4021f2:	83 f9 04             	cmp    $0x4,%ecx
  4021f5:	0f 82 b0 02 00 00    	jb     4024ab <main+0x121b>
  4021fb:	45 85 d2             	test   %r10d,%r10d
  4021fe:	74 54                	je     402254 <main+0xfc4>
  402200:	45 85 db             	test   %r11d,%r11d
  402203:	0f 85 a2 02 00 00    	jne    4024ab <main+0x121b>
  402209:	41 3b cd             	cmp    %r13d,%ecx
  40220c:	0f 82 99 02 00 00    	jb     4024ab <main+0x121b>
  402212:	44 89 e0             	mov    %r12d,%eax
  402215:	41 89 f6             	mov    %esi,%r14d
  402218:	45 33 ff             	xor    %r15d,%r15d
  40221b:	45 85 e4             	test   %r12d,%r12d
  40221e:	74 3f                	je     40225f <main+0xfcf>
  402220:	48 8b 94 24 18 02 00 	mov    0x218(%rsp),%rdx
  402227:	00 
  402228:	c4 a1 7b 10 04 ff    	vmovsd (%rdi,%r15,8),%xmm0
  40222e:	c4 a2 f1 a9 84 fa c0 	vfmadd213sd -0x9c40(%rdx,%r15,8),%xmm1,%xmm0
  402235:	63 ff ff 
  402238:	c4 a1 7b 11 84 fa c0 	vmovsd %xmm0,-0x9c40(%rdx,%r15,8)
  40223f:	63 ff ff 
  402242:	49 ff c7             	inc    %r15
  402245:	4c 3b f8             	cmp    %rax,%r15
  402248:	72 de                	jb     402228 <main+0xf98>
  40224a:	48 8b 94 24 28 02 00 	mov    0x228(%rsp),%rdx
  402251:	00 
  402252:	eb 0b                	jmp    40225f <main+0xfcf>
  402254:	48 8b 84 24 a0 01 00 	mov    0x1a0(%rsp),%rax
  40225b:	00 
  40225c:	41 89 ce             	mov    %ecx,%r14d
  40225f:	c4 e2 7d 19 c1       	vbroadcastsd %xmm1,%ymm0
  402264:	4d 63 f6             	movslq %r14d,%r14
  402267:	4c 8b bc 24 18 02 00 	mov    0x218(%rsp),%r15
  40226e:	00 
  40226f:	c5 fd 10 14 c7       	vmovupd (%rdi,%rax,8),%ymm2
  402274:	c4 c2 fd a8 94 c7 c0 	vfmadd213pd -0x9c40(%r15,%rax,8),%ymm0,%ymm2
  40227b:	63 ff ff 
  40227e:	c4 c1 7d 11 94 c7 c0 	vmovupd %ymm2,-0x9c40(%r15,%rax,8)
  402285:	63 ff ff 
  402288:	48 83 c0 04          	add    $0x4,%rax
  40228c:	49 3b c6             	cmp    %r14,%rax
  40228f:	72 de                	jb     40226f <main+0xfdf>
  402291:	4c 3b f3             	cmp    %rbx,%r14
  402294:	73 2a                	jae    4022c0 <main+0x1030>
  402296:	48 8b 84 24 18 02 00 	mov    0x218(%rsp),%rax
  40229d:	00 
  40229e:	c4 a1 7b 10 04 f7    	vmovsd (%rdi,%r14,8),%xmm0
  4022a4:	c4 a2 f1 a9 84 f0 c0 	vfmadd213sd -0x9c40(%rax,%r14,8),%xmm1,%xmm0
  4022ab:	63 ff ff 
  4022ae:	c4 a1 7b 11 84 f0 c0 	vmovsd %xmm0,-0x9c40(%rax,%r14,8)
  4022b5:	63 ff ff 
  4022b8:	49 ff c6             	inc    %r14
  4022bb:	4c 3b f3             	cmp    %rbx,%r14
  4022be:	72 de                	jb     40229e <main+0x100e>
  4022c0:	49 ff c0             	inc    %r8
  4022c3:	48 81 c7 40 9c 00 00 	add    $0x9c40,%rdi
  4022ca:	4c 3b c2             	cmp    %rdx,%r8
  4022cd:	0f 82 15 ff ff ff    	jb     4021e8 <main+0xf58>
  4022d3:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  4022d8:	33 db                	xor    %ebx,%ebx
  4022da:	48 ff c6             	inc    %rsi
  4022dd:	48 8b 54 24 58       	mov    0x58(%rsp),%rdx
  4022e2:	48 81 c2 40 9c 00 00 	add    $0x9c40,%rdx
  4022e9:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
  4022ee:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  4022f3:	48 3b 74 24 28       	cmp    0x28(%rsp),%rsi
  4022f8:	0f 82 79 fe ff ff    	jb     402177 <main+0xee7>
  4022fe:	4c 8b 94 24 a8 01 00 	mov    0x1a8(%rsp),%r10
  402305:	00 
  402306:	41 bb 80 00 00 00    	mov    $0x80,%r11d
  40230c:	44 8b 4c 24 10       	mov    0x10(%rsp),%r9d
  402311:	44 8b 04 24          	mov    (%rsp),%r8d
  402315:	48 8b b4 24 98 02 00 	mov    0x298(%rsp),%rsi
  40231c:	00 
  40231d:	48 8b 9c 24 00 02 00 	mov    0x200(%rsp),%rbx
  402324:	00 
  402325:	49 ff c2             	inc    %r10
  402328:	49 83 fa 28          	cmp    $0x28,%r10
  40232c:	0f 82 45 f2 ff ff    	jb     401577 <main+0x2e7>
  402332:	8b 84 24 90 02 00 00 	mov    0x290(%rsp),%eax
  402339:	45 33 f6             	xor    %r14d,%r14d
  40233c:	ff c0                	inc    %eax
  40233e:	48 89 b4 24 98 02 00 	mov    %rsi,0x298(%rsp)
  402345:	00 
  402346:	8b b4 24 88 02 00 00 	mov    0x288(%rsp),%esi
  40234d:	4c 8b bc 24 80 02 00 	mov    0x280(%rsp),%r15
  402354:	00 
  402355:	83 f8 28             	cmp    $0x28,%eax
  402358:	0f 82 7d f1 ff ff    	jb     4014db <main+0x24b>
  40235e:	ff c6                	inc    %esi
  402360:	83 fe 28             	cmp    $0x28,%esi
  402363:	0f 82 4d f1 ff ff    	jb     4014b6 <main+0x226>
  402369:	c5 f8 77             	vzeroupper
  40236c:	4c 8b 64 24 08       	mov    0x8(%rsp),%r12
  402371:	4c 8b ac 24 98 02 00 	mov    0x298(%rsp),%r13
  402378:	00 
  402379:	e8 e2 ec ff ff       	call   401060 <clock@plt>
  40237e:	49 2b c4             	sub    %r12,%rax
  402381:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  402385:	c4 e1 fb 2a c0       	vcvtsi2sd %rax,%xmm0,%xmm0
  40238a:	be 80 50 40 00       	mov    $0x405080,%esi
  40238f:	ba 88 13 00 00       	mov    $0x1388,%edx
  402394:	b8 01 00 00 00       	mov    $0x1,%eax
  402399:	c5 fb 5e 05 cf 2c 00 	vdivsd 0x2ccf(%rip),%xmm0,%xmm0        # 405070 <_IO_stdin_used+0x70>
  4023a0:	00 
  4023a1:	48 8b 3d 18 5d 00 00 	mov    0x5d18(%rip),%rdi        # 4080c0 <stderr@GLIBC_2.2.5>
  4023a8:	e8 23 ed ff ff       	call   4010d0 <fprintf@plt>
  4023ad:	bf 9c 50 40 00       	mov    $0x40509c,%edi
  4023b2:	be a8 50 40 00       	mov    $0x4050a8,%esi
  4023b7:	e8 74 ed ff ff       	call   401130 <fopen@plt>
  4023bc:	49 89 c4             	mov    %rax,%r12
  4023bf:	4d 85 e4             	test   %r12,%r12
  4023c2:	0f 84 eb 00 00 00    	je     4024b3 <main+0x1223>
  4023c8:	4c 89 e7             	mov    %r12,%rdi
  4023cb:	be ac 50 40 00       	mov    $0x4050ac,%esi
  4023d0:	ba 88 13 00 00       	mov    $0x1388,%edx
  4023d5:	33 c0                	xor    %eax,%eax
  4023d7:	e8 f4 ec ff ff       	call   4010d0 <fprintf@plt>
  4023dc:	33 d2                	xor    %edx,%edx
  4023de:	4c 89 e8             	mov    %r13,%rax
  4023e1:	4c 89 ac 24 98 02 00 	mov    %r13,0x298(%rsp)
  4023e8:	00 
  4023e9:	41 89 d5             	mov    %edx,%r13d
  4023ec:	48 89 9c 24 00 02 00 	mov    %rbx,0x200(%rsp)
  4023f3:	00 
  4023f4:	48 89 c3             	mov    %rax,%rbx
  4023f7:	4c 89 bc 24 80 02 00 	mov    %r15,0x280(%rsp)
  4023fe:	00 
  4023ff:	4d 89 f7             	mov    %r14,%r15
  402402:	4c 89 e7             	mov    %r12,%rdi
  402405:	be b4 50 40 00       	mov    $0x4050b4,%esi
  40240a:	b8 01 00 00 00       	mov    $0x1,%eax
  40240f:	c4 a1 7b 10 04 fb    	vmovsd (%rbx,%r15,8),%xmm0
  402415:	e8 b6 ec ff ff       	call   4010d0 <fprintf@plt>
  40241a:	49 ff c7             	inc    %r15
  40241d:	49 81 ff e8 03 00 00 	cmp    $0x3e8,%r15
  402424:	7c dc                	jl     402402 <main+0x1172>
  402426:	bf 0a 00 00 00       	mov    $0xa,%edi
  40242b:	4c 89 e6             	mov    %r12,%rsi
  40242e:	e8 8d ec ff ff       	call   4010c0 <fputc@plt>
  402433:	41 ff c5             	inc    %r13d
  402436:	48 81 c3 40 9c 00 00 	add    $0x9c40,%rbx
  40243d:	41 81 fd e8 03 00 00 	cmp    $0x3e8,%r13d
  402444:	7c b9                	jl     4023ff <main+0x116f>
  402446:	4c 89 e7             	mov    %r12,%rdi
  402449:	4c 8b ac 24 98 02 00 	mov    0x298(%rsp),%r13
  402450:	00 
  402451:	48 8b 9c 24 00 02 00 	mov    0x200(%rsp),%rbx
  402458:	00 
  402459:	4c 8b bc 24 80 02 00 	mov    0x280(%rsp),%r15
  402460:	00 
  402461:	e8 0a ec ff ff       	call   401070 <fclose@plt>
  402466:	4c 89 ff             	mov    %r15,%rdi
  402469:	e8 d2 eb ff ff       	call   401040 <free@plt>
  40246e:	48 89 df             	mov    %rbx,%rdi
  402471:	e8 ca eb ff ff       	call   401040 <free@plt>
  402476:	4c 89 ef             	mov    %r13,%rdi
  402479:	e8 c2 eb ff ff       	call   401040 <free@plt>
  40247e:	33 c0                	xor    %eax,%eax
  402480:	48 81 c4 d8 03 00 00 	add    $0x3d8,%rsp
  402487:	5b                   	pop    %rbx
  402488:	41 5f                	pop    %r15
  40248a:	41 5e                	pop    %r14
  40248c:	41 5d                	pop    %r13
  40248e:	41 5c                	pop    %r12
  402490:	48 89 ec             	mov    %rbp,%rsp
  402493:	5d                   	pop    %rbp
  402494:	c3                   	ret
  402495:	45 33 e4             	xor    %r12d,%r12d
  402498:	e9 59 ef ff ff       	jmp    4013f6 <main+0x166>
  40249d:	33 f6                	xor    %esi,%esi
  40249f:	e9 70 f7 ff ff       	jmp    401c14 <main+0x984>
  4024a4:	33 db                	xor    %ebx,%ebx
  4024a6:	e9 24 fb ff ff       	jmp    401fcf <main+0xd3f>
  4024ab:	45 33 f6             	xor    %r14d,%r14d
  4024ae:	e9 de fd ff ff       	jmp    402291 <main+0x1001>
  4024b3:	bf bc 50 40 00       	mov    $0x4050bc,%edi
  4024b8:	e8 83 ec ff ff       	call   401140 <perror@plt>
  4024bd:	b8 01 00 00 00       	mov    $0x1,%eax
  4024c2:	48 81 c4 d8 03 00 00 	add    $0x3d8,%rsp
  4024c9:	5b                   	pop    %rbx
  4024ca:	41 5f                	pop    %r15
  4024cc:	41 5e                	pop    %r14
  4024ce:	41 5d                	pop    %r13
  4024d0:	41 5c                	pop    %r12
  4024d2:	48 89 ec             	mov    %rbp,%rsp
  4024d5:	5d                   	pop    %rbp
  4024d6:	c3                   	ret
  4024d7:	45 33 d2             	xor    %r10d,%r10d
  4024da:	e9 64 ef ff ff       	jmp    401443 <main+0x1b3>
  4024df:	90                   	nop

00000000004024e0 <__intel_new_feature_proc_init_n>:
  4024e0:	f3 0f 1e fa          	endbr64
  4024e4:	55                   	push   %rbp
  4024e5:	41 57                	push   %r15
  4024e7:	41 56                	push   %r14
  4024e9:	41 55                	push   %r13
  4024eb:	41 54                	push   %r12
  4024ed:	53                   	push   %rbx
  4024ee:	48 81 ec 38 04 00 00 	sub    $0x438,%rsp
  4024f5:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4024fc:	00 00 
  4024fe:	48 89 84 24 30 04 00 	mov    %rax,0x430(%rsp)
  402505:	00 
  402506:	0f 57 c0             	xorps  %xmm0,%xmm0
  402509:	0f 29 44 24 10       	movaps %xmm0,0x10(%rsp)
  40250e:	48 c7 c1 d0 80 40 00 	mov    $0x4080d0,%rcx
  402515:	48 83 39 00          	cmpq   $0x0,(%rcx)
  402519:	75 17                	jne    402532 <__intel_new_feature_proc_init_n+0x52>
  40251b:	e8 80 04 00 00       	call   4029a0 <__intel_cpu_features_init>
  402520:	85 c0                	test   %eax,%eax
  402522:	0f 85 0b 02 00 00    	jne    402733 <__intel_new_feature_proc_init_n+0x253>
  402528:	48 83 39 00          	cmpq   $0x0,(%rcx)
  40252c:	0f 84 01 02 00 00    	je     402733 <__intel_new_feature_proc_init_n+0x253>
  402532:	83 ff 02             	cmp    $0x2,%edi
  402535:	7d 38                	jge    40256f <__intel_new_feature_proc_init_n+0x8f>
  402537:	48 63 c7             	movslq %edi,%rax
  40253a:	48 8b 0c c1          	mov    (%rcx,%rax,8),%rcx
  40253e:	48 f7 d1             	not    %rcx
  402541:	48 85 ce             	test   %rcx,%rsi
  402544:	75 48                	jne    40258e <__intel_new_feature_proc_init_n+0xae>
  402546:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40254d:	00 00 
  40254f:	48 3b 84 24 30 04 00 	cmp    0x430(%rsp),%rax
  402556:	00 
  402557:	0f 85 e8 02 00 00    	jne    402845 <__intel_new_feature_proc_init_n+0x365>
  40255d:	48 81 c4 38 04 00 00 	add    $0x438,%rsp
  402564:	5b                   	pop    %rbx
  402565:	41 5c                	pop    %r12
  402567:	41 5d                	pop    %r13
  402569:	41 5e                	pop    %r14
  40256b:	41 5f                	pop    %r15
  40256d:	5d                   	pop    %rbp
  40256e:	c3                   	ret
  40256f:	bf 01 00 00 00       	mov    $0x1,%edi
  402574:	31 f6                	xor    %esi,%esi
  402576:	31 d2                	xor    %edx,%edx
  402578:	31 c0                	xor    %eax,%eax
  40257a:	e8 b1 1e 00 00       	call   404430 <__libirc_print>
  40257f:	bf 01 00 00 00       	mov    $0x1,%edi
  402584:	be 3a 00 00 00       	mov    $0x3a,%esi
  402589:	e9 bf 01 00 00       	jmp    40274d <__intel_new_feature_proc_init_n+0x26d>
  40258e:	48 21 f1             	and    %rsi,%rcx
  402591:	48 89 4c c4 10       	mov    %rcx,0x10(%rsp,%rax,8)
  402596:	45 31 ff             	xor    %r15d,%r15d
  402599:	bf 39 00 00 00       	mov    $0x39,%edi
  40259e:	31 f6                	xor    %esi,%esi
  4025a0:	31 c0                	xor    %eax,%eax
  4025a2:	e8 49 1c 00 00       	call   4041f0 <__libirc_get_msg>
  4025a7:	48 89 04 24          	mov    %rax,(%rsp)
  4025ab:	c6 44 24 30 00       	movb   $0x0,0x30(%rsp)
  4025b0:	bd 01 00 00 00       	mov    $0x1,%ebp
  4025b5:	4c 8d 64 24 10       	lea    0x10(%rsp),%r12
  4025ba:	4c 8d 6c 24 30       	lea    0x30(%rsp),%r13
  4025bf:	31 db                	xor    %ebx,%ebx
  4025c1:	eb 31                	jmp    4025f4 <__intel_new_feature_proc_init_n+0x114>
  4025c3:	b8 ff 03 00 00       	mov    $0x3ff,%eax
  4025c8:	44 29 f8             	sub    %r15d,%eax
  4025cb:	48 63 d0             	movslq %eax,%rdx
  4025ce:	b9 00 04 00 00       	mov    $0x400,%ecx
  4025d3:	4c 89 ef             	mov    %r13,%rdi
  4025d6:	4c 89 f6             	mov    %r14,%rsi
  4025d9:	e8 a2 eb ff ff       	call   401180 <__strncat_chk@plt>
  4025de:	4c 89 ef             	mov    %r13,%rdi
  4025e1:	e8 9a ea ff ff       	call   401080 <strlen@plt>
  4025e6:	49 89 c7             	mov    %rax,%r15
  4025e9:	ff c5                	inc    %ebp
  4025eb:	83 fd 47             	cmp    $0x47,%ebp
  4025ee:	0f 84 26 01 00 00    	je     40271a <__intel_new_feature_proc_init_n+0x23a>
  4025f4:	89 e8                	mov    %ebp,%eax
  4025f6:	e8 95 19 00 00       	call   403f90 <__libirc_get_feature_bitpos>
  4025fb:	85 c0                	test   %eax,%eax
  4025fd:	78 ea                	js     4025e9 <__intel_new_feature_proc_init_n+0x109>
  4025ff:	4c 89 e7             	mov    %r12,%rdi
  402602:	89 ee                	mov    %ebp,%esi
  402604:	e8 c7 19 00 00       	call   403fd0 <__libirc_get_cpu_feature>
  402609:	85 c0                	test   %eax,%eax
  40260b:	74 dc                	je     4025e9 <__intel_new_feature_proc_init_n+0x109>
  40260d:	4c 89 e7             	mov    %r12,%rdi
  402610:	89 ee                	mov    %ebp,%esi
  402612:	e8 b9 19 00 00       	call   403fd0 <__libirc_get_cpu_feature>
  402617:	85 c0                	test   %eax,%eax
  402619:	0f 88 e6 00 00 00    	js     402705 <__intel_new_feature_proc_init_n+0x225>
  40261f:	89 ef                	mov    %ebp,%edi
  402621:	e8 7a 0e 00 00       	call   4034a0 <__libirc_get_feature_name>
  402626:	48 85 c0             	test   %rax,%rax
  402629:	0f 84 d6 00 00 00    	je     402705 <__intel_new_feature_proc_init_n+0x225>
  40262f:	49 89 c6             	mov    %rax,%r14
  402632:	80 38 00             	cmpb   $0x0,(%rax)
  402635:	0f 84 ca 00 00 00    	je     402705 <__intel_new_feature_proc_init_n+0x225>
  40263b:	80 7c 24 30 00       	cmpb   $0x0,0x30(%rsp)
  402640:	74 81                	je     4025c3 <__intel_new_feature_proc_init_n+0xe3>
  402642:	48 85 db             	test   %rbx,%rbx
  402645:	0f 84 b2 00 00 00    	je     4026fd <__intel_new_feature_proc_init_n+0x21d>
  40264b:	4d 89 ec             	mov    %r13,%r12
  40264e:	48 89 df             	mov    %rbx,%rdi
  402651:	e8 2a ea ff ff       	call   401080 <strlen@plt>
  402656:	49 89 c5             	mov    %rax,%r13
  402659:	48 8d 3d 64 2a 00 00 	lea    0x2a64(%rip),%rdi        # 4050c4 <_IO_stdin_used+0xc4>
  402660:	e8 1b ea ff ff       	call   401080 <strlen@plt>
  402665:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  40266a:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  40266f:	49 63 df             	movslq %r15d,%rbx
  402672:	48 8b 3c 24          	mov    (%rsp),%rdi
  402676:	e8 05 ea ff ff       	call   401080 <strlen@plt>
  40267b:	49 89 c7             	mov    %rax,%r15
  40267e:	4c 89 f7             	mov    %r14,%rdi
  402681:	e8 fa e9 ff ff       	call   401080 <strlen@plt>
  402686:	49 01 dd             	add    %rbx,%r13
  402689:	4c 03 6c 24 28       	add    0x28(%rsp),%r13
  40268e:	4c 01 f8             	add    %r15,%rax
  402691:	4c 01 e8             	add    %r13,%rax
  402694:	b9 ff 03 00 00       	mov    $0x3ff,%ecx
  402699:	29 d9                	sub    %ebx,%ecx
  40269b:	48 63 d1             	movslq %ecx,%rdx
  40269e:	48 3d ff 03 00 00    	cmp    $0x3ff,%rax
  4026a4:	0f 87 dd 00 00 00    	ja     402787 <__intel_new_feature_proc_init_n+0x2a7>
  4026aa:	b9 00 04 00 00       	mov    $0x400,%ecx
  4026af:	4d 89 e5             	mov    %r12,%r13
  4026b2:	4c 89 e7             	mov    %r12,%rdi
  4026b5:	48 8d 35 08 2a 00 00 	lea    0x2a08(%rip),%rsi        # 4050c4 <_IO_stdin_used+0xc4>
  4026bc:	e8 bf ea ff ff       	call   401180 <__strncat_chk@plt>
  4026c1:	4c 89 e7             	mov    %r12,%rdi
  4026c4:	e8 b7 e9 ff ff       	call   401080 <strlen@plt>
  4026c9:	48 c1 e0 20          	shl    $0x20,%rax
  4026cd:	48 ba 00 00 00 00 ff 	movabs $0x3ff00000000,%rdx
  4026d4:	03 00 00 
  4026d7:	48 29 c2             	sub    %rax,%rdx
  4026da:	48 c1 fa 20          	sar    $0x20,%rdx
  4026de:	b9 00 04 00 00       	mov    $0x400,%ecx
  4026e3:	4c 89 e7             	mov    %r12,%rdi
  4026e6:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  4026eb:	e8 90 ea ff ff       	call   401180 <__strncat_chk@plt>
  4026f0:	4c 89 f3             	mov    %r14,%rbx
  4026f3:	4c 8d 64 24 10       	lea    0x10(%rsp),%r12
  4026f8:	e9 e1 fe ff ff       	jmp    4025de <__intel_new_feature_proc_init_n+0xfe>
  4026fd:	4c 89 f3             	mov    %r14,%rbx
  402700:	e9 e4 fe ff ff       	jmp    4025e9 <__intel_new_feature_proc_init_n+0x109>
  402705:	bf 01 00 00 00       	mov    $0x1,%edi
  40270a:	31 f6                	xor    %esi,%esi
  40270c:	31 d2                	xor    %edx,%edx
  40270e:	31 c0                	xor    %eax,%eax
  402710:	e8 1b 1d 00 00       	call   404430 <__libirc_print>
  402715:	e9 ce 00 00 00       	jmp    4027e8 <__intel_new_feature_proc_init_n+0x308>
  40271a:	48 85 db             	test   %rbx,%rbx
  40271d:	0f 84 ac 00 00 00    	je     4027cf <__intel_new_feature_proc_init_n+0x2ef>
  402723:	49 89 dc             	mov    %rbx,%r12
  402726:	b8 ff 03 00 00       	mov    $0x3ff,%eax
  40272b:	44 29 f8             	sub    %r15d,%eax
  40272e:	48 63 d0             	movslq %eax,%rdx
  402731:	eb 59                	jmp    40278c <__intel_new_feature_proc_init_n+0x2ac>
  402733:	bf 01 00 00 00       	mov    $0x1,%edi
  402738:	31 f6                	xor    %esi,%esi
  40273a:	31 d2                	xor    %edx,%edx
  40273c:	31 c0                	xor    %eax,%eax
  40273e:	e8 ed 1c 00 00       	call   404430 <__libirc_print>
  402743:	bf 01 00 00 00       	mov    $0x1,%edi
  402748:	be 3b 00 00 00       	mov    $0x3b,%esi
  40274d:	31 d2                	xor    %edx,%edx
  40274f:	31 c0                	xor    %eax,%eax
  402751:	e8 da 1c 00 00       	call   404430 <__libirc_print>
  402756:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40275d:	00 00 
  40275f:	48 3b 84 24 30 04 00 	cmp    0x430(%rsp),%rax
  402766:	00 
  402767:	0f 85 d8 00 00 00    	jne    402845 <__intel_new_feature_proc_init_n+0x365>
  40276d:	bf 01 00 00 00       	mov    $0x1,%edi
  402772:	31 f6                	xor    %esi,%esi
  402774:	31 d2                	xor    %edx,%edx
  402776:	31 c0                	xor    %eax,%eax
  402778:	e8 b3 1c 00 00       	call   404430 <__libirc_print>
  40277d:	bf 01 00 00 00       	mov    $0x1,%edi
  402782:	e8 d9 e9 ff ff       	call   401160 <exit@plt>
  402787:	4c 8b 64 24 08       	mov    0x8(%rsp),%r12
  40278c:	4c 8d 74 24 30       	lea    0x30(%rsp),%r14
  402791:	b9 00 04 00 00       	mov    $0x400,%ecx
  402796:	4c 89 f7             	mov    %r14,%rdi
  402799:	48 8b 34 24          	mov    (%rsp),%rsi
  40279d:	e8 de e9 ff ff       	call   401180 <__strncat_chk@plt>
  4027a2:	4c 89 f7             	mov    %r14,%rdi
  4027a5:	e8 d6 e8 ff ff       	call   401080 <strlen@plt>
  4027aa:	48 c1 e0 20          	shl    $0x20,%rax
  4027ae:	48 ba 00 00 00 00 ff 	movabs $0x3ff00000000,%rdx
  4027b5:	03 00 00 
  4027b8:	48 29 c2             	sub    %rax,%rdx
  4027bb:	48 c1 fa 20          	sar    $0x20,%rdx
  4027bf:	b9 00 04 00 00       	mov    $0x400,%ecx
  4027c4:	4c 89 f7             	mov    %r14,%rdi
  4027c7:	4c 89 e6             	mov    %r12,%rsi
  4027ca:	e8 b1 e9 ff ff       	call   401180 <__strncat_chk@plt>
  4027cf:	0f b6 5c 24 30       	movzbl 0x30(%rsp),%ebx
  4027d4:	bf 01 00 00 00       	mov    $0x1,%edi
  4027d9:	31 f6                	xor    %esi,%esi
  4027db:	31 d2                	xor    %edx,%edx
  4027dd:	31 c0                	xor    %eax,%eax
  4027df:	e8 4c 1c 00 00       	call   404430 <__libirc_print>
  4027e4:	84 db                	test   %bl,%bl
  4027e6:	75 15                	jne    4027fd <__intel_new_feature_proc_init_n+0x31d>
  4027e8:	bf 01 00 00 00       	mov    $0x1,%edi
  4027ed:	be 3a 00 00 00       	mov    $0x3a,%esi
  4027f2:	31 d2                	xor    %edx,%edx
  4027f4:	31 c0                	xor    %eax,%eax
  4027f6:	e8 35 1c 00 00       	call   404430 <__libirc_print>
  4027fb:	eb 1b                	jmp    402818 <__intel_new_feature_proc_init_n+0x338>
  4027fd:	48 8d 4c 24 30       	lea    0x30(%rsp),%rcx
  402802:	bf 01 00 00 00       	mov    $0x1,%edi
  402807:	be 38 00 00 00       	mov    $0x38,%esi
  40280c:	ba 01 00 00 00       	mov    $0x1,%edx
  402811:	31 c0                	xor    %eax,%eax
  402813:	e8 18 1c 00 00       	call   404430 <__libirc_print>
  402818:	bf 01 00 00 00       	mov    $0x1,%edi
  40281d:	31 f6                	xor    %esi,%esi
  40281f:	31 d2                	xor    %edx,%edx
  402821:	31 c0                	xor    %eax,%eax
  402823:	e8 08 1c 00 00       	call   404430 <__libirc_print>
  402828:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40282f:	00 00 
  402831:	48 3b 84 24 30 04 00 	cmp    0x430(%rsp),%rax
  402838:	00 
  402839:	75 0a                	jne    402845 <__intel_new_feature_proc_init_n+0x365>
  40283b:	bf 01 00 00 00       	mov    $0x1,%edi
  402840:	e8 1b e9 ff ff       	call   401160 <exit@plt>
  402845:	e8 46 e8 ff ff       	call   401090 <__stack_chk_fail@plt>
  40284a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000402850 <__intel_new_feature_proc_init>:
  402850:	f3 0f 1e fa          	endbr64
  402854:	53                   	push   %rbx
  402855:	89 fb                	mov    %edi,%ebx
  402857:	31 ff                	xor    %edi,%edi
  402859:	e8 82 fc ff ff       	call   4024e0 <__intel_new_feature_proc_init_n>
  40285e:	48 c7 c7 d0 80 40 00 	mov    $0x4080d0,%rdi
  402865:	be 06 00 00 00       	mov    $0x6,%esi
  40286a:	e8 61 17 00 00       	call   403fd0 <__libirc_get_cpu_feature>
  40286f:	83 f8 01             	cmp    $0x1,%eax
  402872:	75 0a                	jne    40287e <__intel_new_feature_proc_init+0x2e>
  402874:	31 ff                	xor    %edi,%edi
  402876:	89 de                	mov    %ebx,%esi
  402878:	5b                   	pop    %rbx
  402879:	e9 52 00 00 00       	jmp    4028d0 <__intel_proc_init_ftzdazule>
  40287e:	85 c0                	test   %eax,%eax
  402880:	78 02                	js     402884 <__intel_new_feature_proc_init+0x34>
  402882:	5b                   	pop    %rbx
  402883:	c3                   	ret
  402884:	bf 01 00 00 00       	mov    $0x1,%edi
  402889:	31 f6                	xor    %esi,%esi
  40288b:	31 d2                	xor    %edx,%edx
  40288d:	31 c0                	xor    %eax,%eax
  40288f:	e8 9c 1b 00 00       	call   404430 <__libirc_print>
  402894:	bf 01 00 00 00       	mov    $0x1,%edi
  402899:	be 3a 00 00 00       	mov    $0x3a,%esi
  40289e:	31 d2                	xor    %edx,%edx
  4028a0:	31 c0                	xor    %eax,%eax
  4028a2:	e8 89 1b 00 00       	call   404430 <__libirc_print>
  4028a7:	bf 01 00 00 00       	mov    $0x1,%edi
  4028ac:	31 f6                	xor    %esi,%esi
  4028ae:	31 d2                	xor    %edx,%edx
  4028b0:	31 c0                	xor    %eax,%eax
  4028b2:	e8 79 1b 00 00       	call   404430 <__libirc_print>
  4028b7:	bf 01 00 00 00       	mov    $0x1,%edi
  4028bc:	e8 9f e8 ff ff       	call   401160 <exit@plt>
  4028c1:	0f 1f 00             	nopl   (%rax)
  4028c4:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4028cb:	00 00 00 
  4028ce:	66 90                	xchg   %ax,%ax

00000000004028d0 <__intel_proc_init_ftzdazule>:
  4028d0:	f3 0f 1e fa          	endbr64
  4028d4:	55                   	push   %rbp
  4028d5:	41 56                	push   %r14
  4028d7:	53                   	push   %rbx
  4028d8:	48 81 ec 20 02 00 00 	sub    $0x220,%rsp
  4028df:	89 f3                	mov    %esi,%ebx
  4028e1:	41 89 f6             	mov    %esi,%r14d
  4028e4:	41 83 e6 04          	and    $0x4,%r14d
  4028e8:	89 f5                	mov    %esi,%ebp
  4028ea:	83 e5 02             	and    $0x2,%ebp
  4028ed:	74 07                	je     4028f6 <__intel_proc_init_ftzdazule+0x26>
  4028ef:	89 f8                	mov    %edi,%eax
  4028f1:	83 e0 02             	and    $0x2,%eax
  4028f4:	74 12                	je     402908 <__intel_proc_init_ftzdazule+0x38>
  4028f6:	31 c0                	xor    %eax,%eax
  4028f8:	45 85 f6             	test   %r14d,%r14d
  4028fb:	74 38                	je     402935 <__intel_proc_init_ftzdazule+0x65>
  4028fd:	b8 01 00 00 00       	mov    $0x1,%eax
  402902:	40 f6 c7 04          	test   $0x4,%dil
  402906:	75 2d                	jne    402935 <__intel_proc_init_ftzdazule+0x65>
  402908:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
  40290d:	ba 00 02 00 00       	mov    $0x200,%edx
  402912:	31 f6                	xor    %esi,%esi
  402914:	e8 97 e7 ff ff       	call   4010b0 <memset@plt>
  402919:	0f ae 44 24 20       	fxsave 0x20(%rsp)
  40291e:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
  402922:	89 c1                	mov    %eax,%ecx
  402924:	c1 e1 19             	shl    $0x19,%ecx
  402927:	c1 f9 1f             	sar    $0x1f,%ecx
  40292a:	21 cd                	and    %ecx,%ebp
  40292c:	c1 e0 0e             	shl    $0xe,%eax
  40292f:	c1 f8 1f             	sar    $0x1f,%eax
  402932:	44 21 f0             	and    %r14d,%eax
  402935:	f6 c3 01             	test   $0x1,%bl
  402938:	74 17                	je     402951 <__intel_proc_init_ftzdazule+0x81>
  40293a:	0f ae 5c 24 1c       	stmxcsr 0x1c(%rsp)
  40293f:	b9 00 80 00 00       	mov    $0x8000,%ecx
  402944:	0b 4c 24 1c          	or     0x1c(%rsp),%ecx
  402948:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
  40294c:	0f ae 54 24 18       	ldmxcsr 0x18(%rsp)
  402951:	85 ed                	test   %ebp,%ebp
  402953:	74 15                	je     40296a <__intel_proc_init_ftzdazule+0x9a>
  402955:	0f ae 5c 24 14       	stmxcsr 0x14(%rsp)
  40295a:	8b 4c 24 14          	mov    0x14(%rsp),%ecx
  40295e:	83 c9 40             	or     $0x40,%ecx
  402961:	89 4c 24 10          	mov    %ecx,0x10(%rsp)
  402965:	0f ae 54 24 10       	ldmxcsr 0x10(%rsp)
  40296a:	85 c0                	test   %eax,%eax
  40296c:	74 17                	je     402985 <__intel_proc_init_ftzdazule+0xb5>
  40296e:	0f ae 5c 24 0c       	stmxcsr 0xc(%rsp)
  402973:	b8 00 00 02 00       	mov    $0x20000,%eax
  402978:	0b 44 24 0c          	or     0xc(%rsp),%eax
  40297c:	89 44 24 08          	mov    %eax,0x8(%rsp)
  402980:	0f ae 54 24 08       	ldmxcsr 0x8(%rsp)
  402985:	48 81 c4 20 02 00 00 	add    $0x220,%rsp
  40298c:	5b                   	pop    %rbx
  40298d:	41 5e                	pop    %r14
  40298f:	5d                   	pop    %rbp
  402990:	c3                   	ret
  402991:	0f 1f 00             	nopl   (%rax)
  402994:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  40299b:	00 00 00 
  40299e:	66 90                	xchg   %ax,%ax

00000000004029a0 <__intel_cpu_features_init>:
  4029a0:	f3 0f 1e fa          	endbr64
  4029a4:	50                   	push   %rax
  4029a5:	b8 01 00 00 00       	mov    $0x1,%eax
  4029aa:	e8 11 00 00 00       	call   4029c0 <__intel_cpu_features_init_body>
  4029af:	48 83 c4 08          	add    $0x8,%rsp
  4029b3:	c3                   	ret
  4029b4:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4029bb:	00 00 00 
  4029be:	66 90                	xchg   %ax,%ax

00000000004029c0 <__intel_cpu_features_init_body>:
  4029c0:	41 53                	push   %r11
  4029c2:	41 52                	push   %r10
  4029c4:	41 51                	push   %r9
  4029c6:	41 50                	push   %r8
  4029c8:	52                   	push   %rdx
  4029c9:	51                   	push   %rcx
  4029ca:	56                   	push   %rsi
  4029cb:	57                   	push   %rdi
  4029cc:	55                   	push   %rbp
  4029cd:	53                   	push   %rbx
  4029ce:	48 81 ec 38 01 00 00 	sub    $0x138,%rsp
  4029d5:	44 0f 29 bc 24 20 01 	movaps %xmm15,0x120(%rsp)
  4029dc:	00 00 
  4029de:	44 0f 29 b4 24 10 01 	movaps %xmm14,0x110(%rsp)
  4029e5:	00 00 
  4029e7:	44 0f 29 ac 24 00 01 	movaps %xmm13,0x100(%rsp)
  4029ee:	00 00 
  4029f0:	44 0f 29 a4 24 f0 00 	movaps %xmm12,0xf0(%rsp)
  4029f7:	00 00 
  4029f9:	44 0f 29 9c 24 e0 00 	movaps %xmm11,0xe0(%rsp)
  402a00:	00 00 
  402a02:	44 0f 29 94 24 d0 00 	movaps %xmm10,0xd0(%rsp)
  402a09:	00 00 
  402a0b:	44 0f 29 8c 24 c0 00 	movaps %xmm9,0xc0(%rsp)
  402a12:	00 00 
  402a14:	44 0f 29 84 24 b0 00 	movaps %xmm8,0xb0(%rsp)
  402a1b:	00 00 
  402a1d:	0f 29 bc 24 a0 00 00 	movaps %xmm7,0xa0(%rsp)
  402a24:	00 
  402a25:	0f 29 b4 24 90 00 00 	movaps %xmm6,0x90(%rsp)
  402a2c:	00 
  402a2d:	0f 29 ac 24 80 00 00 	movaps %xmm5,0x80(%rsp)
  402a34:	00 
  402a35:	0f 29 64 24 70       	movaps %xmm4,0x70(%rsp)
  402a3a:	0f 29 5c 24 60       	movaps %xmm3,0x60(%rsp)
  402a3f:	0f 29 54 24 50       	movaps %xmm2,0x50(%rsp)
  402a44:	0f 29 4c 24 40       	movaps %xmm1,0x40(%rsp)
  402a49:	0f 29 44 24 30       	movaps %xmm0,0x30(%rsp)
  402a4e:	89 c5                	mov    %eax,%ebp
  402a50:	0f 57 c0             	xorps  %xmm0,%xmm0
  402a53:	0f 29 04 24          	movaps %xmm0,(%rsp)
  402a57:	0f 29 44 24 20       	movaps %xmm0,0x20(%rsp)
  402a5c:	48 89 e0             	mov    %rsp,%rax
  402a5f:	b9 01 00 00 00       	mov    $0x1,%ecx
  402a64:	e8 b7 15 00 00       	call   404020 <__libirc_set_cpu_feature>
  402a69:	85 c0                	test   %eax,%eax
  402a6b:	0f 85 81 03 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402a71:	31 c0                	xor    %eax,%eax
  402a73:	0f a2                	cpuid
  402a75:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  402a79:	89 5c 24 18          	mov    %ebx,0x18(%rsp)
  402a7d:	89 4c 24 14          	mov    %ecx,0x14(%rsp)
  402a81:	89 54 24 10          	mov    %edx,0x10(%rsp)
  402a85:	83 7c 24 1c 00       	cmpl   $0x0,0x1c(%rsp)
  402a8a:	0f 84 55 03 00 00    	je     402de5 <__intel_cpu_features_init_body+0x425>
  402a90:	83 fd 01             	cmp    $0x1,%ebp
  402a93:	75 2a                	jne    402abf <__intel_cpu_features_init_body+0xff>
  402a95:	81 7c 24 18 47 65 6e 	cmpl   $0x756e6547,0x18(%rsp)
  402a9c:	75 
  402a9d:	0f 85 42 03 00 00    	jne    402de5 <__intel_cpu_features_init_body+0x425>
  402aa3:	81 7c 24 10 69 6e 65 	cmpl   $0x49656e69,0x10(%rsp)
  402aaa:	49 
  402aab:	0f 85 34 03 00 00    	jne    402de5 <__intel_cpu_features_init_body+0x425>
  402ab1:	81 7c 24 14 6e 74 65 	cmpl   $0x6c65746e,0x14(%rsp)
  402ab8:	6c 
  402ab9:	0f 85 26 03 00 00    	jne    402de5 <__intel_cpu_features_init_body+0x425>
  402abf:	b8 01 00 00 00       	mov    $0x1,%eax
  402ac4:	0f a2                	cpuid
  402ac6:	41 89 d2             	mov    %edx,%r10d
  402ac9:	41 89 c8             	mov    %ecx,%r8d
  402acc:	41 f6 c2 01          	test   $0x1,%r10b
  402ad0:	74 15                	je     402ae7 <__intel_cpu_features_init_body+0x127>
  402ad2:	48 89 e0             	mov    %rsp,%rax
  402ad5:	b9 02 00 00 00       	mov    $0x2,%ecx
  402ada:	e8 41 15 00 00       	call   404020 <__libirc_set_cpu_feature>
  402adf:	85 c0                	test   %eax,%eax
  402ae1:	0f 85 0b 03 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402ae7:	66 45 85 d2          	test   %r10w,%r10w
  402aeb:	79 15                	jns    402b02 <__intel_cpu_features_init_body+0x142>
  402aed:	48 89 e0             	mov    %rsp,%rax
  402af0:	b9 03 00 00 00       	mov    $0x3,%ecx
  402af5:	e8 26 15 00 00       	call   404020 <__libirc_set_cpu_feature>
  402afa:	85 c0                	test   %eax,%eax
  402afc:	0f 85 f0 02 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402b02:	41 f7 c2 00 00 80 00 	test   $0x800000,%r10d
  402b09:	74 15                	je     402b20 <__intel_cpu_features_init_body+0x160>
  402b0b:	48 89 e0             	mov    %rsp,%rax
  402b0e:	b9 04 00 00 00       	mov    $0x4,%ecx
  402b13:	e8 08 15 00 00       	call   404020 <__libirc_set_cpu_feature>
  402b18:	85 c0                	test   %eax,%eax
  402b1a:	0f 85 d2 02 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402b20:	41 f7 c2 00 00 00 01 	test   $0x1000000,%r10d
  402b27:	0f 85 54 03 00 00    	jne    402e81 <__intel_cpu_features_init_body+0x4c1>
  402b2d:	41 f7 c0 00 00 00 40 	test   $0x40000000,%r8d
  402b34:	74 15                	je     402b4b <__intel_cpu_features_init_body+0x18b>
  402b36:	48 89 e0             	mov    %rsp,%rax
  402b39:	b9 12 00 00 00       	mov    $0x12,%ecx
  402b3e:	e8 dd 14 00 00       	call   404020 <__libirc_set_cpu_feature>
  402b43:	85 c0                	test   %eax,%eax
  402b45:	0f 85 a7 02 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402b4b:	41 f7 c2 00 00 00 01 	test   $0x1000000,%r10d
  402b52:	75 10                	jne    402b64 <__intel_cpu_features_init_body+0x1a4>
  402b54:	b8 07 00 00 00       	mov    $0x7,%eax
  402b59:	31 c9                	xor    %ecx,%ecx
  402b5b:	0f a2                	cpuid
  402b5d:	89 cf                	mov    %ecx,%edi
  402b5f:	89 d6                	mov    %edx,%esi
  402b61:	41 89 d9             	mov    %ebx,%r9d
  402b64:	44 89 c8             	mov    %r9d,%eax
  402b67:	f7 d0                	not    %eax
  402b69:	a9 08 01 00 00       	test   $0x108,%eax
  402b6e:	75 15                	jne    402b85 <__intel_cpu_features_init_body+0x1c5>
  402b70:	48 89 e0             	mov    %rsp,%rax
  402b73:	b9 14 00 00 00       	mov    $0x14,%ecx
  402b78:	e8 a3 14 00 00       	call   404020 <__libirc_set_cpu_feature>
  402b7d:	85 c0                	test   %eax,%eax
  402b7f:	0f 85 6d 02 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402b85:	41 f6 c1 04          	test   $0x4,%r9b
  402b89:	74 15                	je     402ba0 <__intel_cpu_features_init_body+0x1e0>
  402b8b:	48 89 e0             	mov    %rsp,%rax
  402b8e:	b9 36 00 00 00       	mov    $0x36,%ecx
  402b93:	e8 88 14 00 00       	call   404020 <__libirc_set_cpu_feature>
  402b98:	85 c0                	test   %eax,%eax
  402b9a:	0f 85 52 02 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402ba0:	41 f6 c1 10          	test   $0x10,%r9b
  402ba4:	74 15                	je     402bbb <__intel_cpu_features_init_body+0x1fb>
  402ba6:	48 89 e0             	mov    %rsp,%rax
  402ba9:	b9 16 00 00 00       	mov    $0x16,%ecx
  402bae:	e8 6d 14 00 00       	call   404020 <__libirc_set_cpu_feature>
  402bb3:	85 c0                	test   %eax,%eax
  402bb5:	0f 85 37 02 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402bbb:	41 f7 c1 00 08 00 00 	test   $0x800,%r9d
  402bc2:	74 15                	je     402bd9 <__intel_cpu_features_init_body+0x219>
  402bc4:	48 89 e0             	mov    %rsp,%rax
  402bc7:	b9 17 00 00 00       	mov    $0x17,%ecx
  402bcc:	e8 4f 14 00 00       	call   404020 <__libirc_set_cpu_feature>
  402bd1:	85 c0                	test   %eax,%eax
  402bd3:	0f 85 19 02 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402bd9:	41 f7 c1 00 00 08 00 	test   $0x80000,%r9d
  402be0:	74 15                	je     402bf7 <__intel_cpu_features_init_body+0x237>
  402be2:	48 89 e0             	mov    %rsp,%rax
  402be5:	b9 1d 00 00 00       	mov    $0x1d,%ecx
  402bea:	e8 31 14 00 00       	call   404020 <__libirc_set_cpu_feature>
  402bef:	85 c0                	test   %eax,%eax
  402bf1:	0f 85 fb 01 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402bf7:	41 f7 c1 00 00 04 00 	test   $0x40000,%r9d
  402bfe:	74 15                	je     402c15 <__intel_cpu_features_init_body+0x255>
  402c00:	48 89 e0             	mov    %rsp,%rax
  402c03:	b9 1e 00 00 00       	mov    $0x1e,%ecx
  402c08:	e8 13 14 00 00       	call   404020 <__libirc_set_cpu_feature>
  402c0d:	85 c0                	test   %eax,%eax
  402c0f:	0f 85 dd 01 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402c15:	41 f7 c1 00 00 00 01 	test   $0x1000000,%r9d
  402c1c:	74 15                	je     402c33 <__intel_cpu_features_init_body+0x273>
  402c1e:	48 89 e0             	mov    %rsp,%rax
  402c21:	b9 32 00 00 00       	mov    $0x32,%ecx
  402c26:	e8 f5 13 00 00       	call   404020 <__libirc_set_cpu_feature>
  402c2b:	85 c0                	test   %eax,%eax
  402c2d:	0f 85 bf 01 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402c33:	b8 01 00 00 80       	mov    $0x80000001,%eax
  402c38:	0f a2                	cpuid
  402c3a:	f6 c1 20             	test   $0x20,%cl
  402c3d:	74 15                	je     402c54 <__intel_cpu_features_init_body+0x294>
  402c3f:	48 89 e0             	mov    %rsp,%rax
  402c42:	b9 15 00 00 00       	mov    $0x15,%ecx
  402c47:	e8 d4 13 00 00       	call   404020 <__libirc_set_cpu_feature>
  402c4c:	85 c0                	test   %eax,%eax
  402c4e:	0f 85 9e 01 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402c54:	b8 08 00 00 80       	mov    $0x80000008,%eax
  402c59:	0f a2                	cpuid
  402c5b:	f7 c3 00 02 00 00    	test   $0x200,%ebx
  402c61:	74 15                	je     402c78 <__intel_cpu_features_init_body+0x2b8>
  402c63:	48 89 e0             	mov    %rsp,%rax
  402c66:	b9 37 00 00 00       	mov    $0x37,%ecx
  402c6b:	e8 b0 13 00 00       	call   404020 <__libirc_set_cpu_feature>
  402c70:	85 c0                	test   %eax,%eax
  402c72:	0f 85 7a 01 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402c78:	40 f6 c7 20          	test   $0x20,%dil
  402c7c:	74 15                	je     402c93 <__intel_cpu_features_init_body+0x2d3>
  402c7e:	48 89 e0             	mov    %rsp,%rax
  402c81:	b9 3e 00 00 00       	mov    $0x3e,%ecx
  402c86:	e8 95 13 00 00       	call   404020 <__libirc_set_cpu_feature>
  402c8b:	85 c0                	test   %eax,%eax
  402c8d:	0f 85 5f 01 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402c93:	40 84 ff             	test   %dil,%dil
  402c96:	79 15                	jns    402cad <__intel_cpu_features_init_body+0x2ed>
  402c98:	48 89 e0             	mov    %rsp,%rax
  402c9b:	b9 35 00 00 00       	mov    $0x35,%ecx
  402ca0:	e8 7b 13 00 00       	call   404020 <__libirc_set_cpu_feature>
  402ca5:	85 c0                	test   %eax,%eax
  402ca7:	0f 85 45 01 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402cad:	f7 c7 00 01 00 00    	test   $0x100,%edi
  402cb3:	74 15                	je     402cca <__intel_cpu_features_init_body+0x30a>
  402cb5:	48 89 e0             	mov    %rsp,%rax
  402cb8:	b9 2e 00 00 00       	mov    $0x2e,%ecx
  402cbd:	e8 5e 13 00 00       	call   404020 <__libirc_set_cpu_feature>
  402cc2:	85 c0                	test   %eax,%eax
  402cc4:	0f 85 28 01 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402cca:	f7 c7 00 00 40 00    	test   $0x400000,%edi
  402cd0:	74 15                	je     402ce7 <__intel_cpu_features_init_body+0x327>
  402cd2:	48 89 e0             	mov    %rsp,%rax
  402cd5:	b9 33 00 00 00       	mov    $0x33,%ecx
  402cda:	e8 41 13 00 00       	call   404020 <__libirc_set_cpu_feature>
  402cdf:	85 c0                	test   %eax,%eax
  402ce1:	0f 85 0b 01 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402ce7:	f7 c7 00 00 00 01    	test   $0x1000000,%edi
  402ced:	74 15                	je     402d04 <__intel_cpu_features_init_body+0x344>
  402cef:	48 89 e0             	mov    %rsp,%rax
  402cf2:	b9 3b 00 00 00       	mov    $0x3b,%ecx
  402cf7:	e8 24 13 00 00       	call   404020 <__libirc_set_cpu_feature>
  402cfc:	85 c0                	test   %eax,%eax
  402cfe:	0f 85 ee 00 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402d04:	f7 c7 00 00 00 08    	test   $0x8000000,%edi
  402d0a:	74 15                	je     402d21 <__intel_cpu_features_init_body+0x361>
  402d0c:	48 89 e0             	mov    %rsp,%rax
  402d0f:	b9 3c 00 00 00       	mov    $0x3c,%ecx
  402d14:	e8 07 13 00 00       	call   404020 <__libirc_set_cpu_feature>
  402d19:	85 c0                	test   %eax,%eax
  402d1b:	0f 85 d1 00 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402d21:	f7 c7 00 00 00 10    	test   $0x10000000,%edi
  402d27:	74 15                	je     402d3e <__intel_cpu_features_init_body+0x37e>
  402d29:	48 89 e0             	mov    %rsp,%rax
  402d2c:	b9 3d 00 00 00       	mov    $0x3d,%ecx
  402d31:	e8 ea 12 00 00       	call   404020 <__libirc_set_cpu_feature>
  402d36:	85 c0                	test   %eax,%eax
  402d38:	0f 85 b4 00 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402d3e:	f7 c7 00 00 00 20    	test   $0x20000000,%edi
  402d44:	74 15                	je     402d5b <__intel_cpu_features_init_body+0x39b>
  402d46:	48 89 e0             	mov    %rsp,%rax
  402d49:	b9 40 00 00 00       	mov    $0x40,%ecx
  402d4e:	e8 cd 12 00 00       	call   404020 <__libirc_set_cpu_feature>
  402d53:	85 c0                	test   %eax,%eax
  402d55:	0f 85 97 00 00 00    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402d5b:	f7 c6 00 00 10 00    	test   $0x100000,%esi
  402d61:	74 11                	je     402d74 <__intel_cpu_features_init_body+0x3b4>
  402d63:	48 89 e0             	mov    %rsp,%rax
  402d66:	b9 34 00 00 00       	mov    $0x34,%ecx
  402d6b:	e8 b0 12 00 00       	call   404020 <__libirc_set_cpu_feature>
  402d70:	85 c0                	test   %eax,%eax
  402d72:	75 7e                	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402d74:	f7 c6 00 00 04 00    	test   $0x40000,%esi
  402d7a:	74 11                	je     402d8d <__intel_cpu_features_init_body+0x3cd>
  402d7c:	48 89 e0             	mov    %rsp,%rax
  402d7f:	b9 38 00 00 00       	mov    $0x38,%ecx
  402d84:	e8 97 12 00 00       	call   404020 <__libirc_set_cpu_feature>
  402d89:	85 c0                	test   %eax,%eax
  402d8b:	75 65                	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402d8d:	b8 14 00 00 00       	mov    $0x14,%eax
  402d92:	31 c9                	xor    %ecx,%ecx
  402d94:	0f a2                	cpuid
  402d96:	f6 c3 10             	test   $0x10,%bl
  402d99:	74 11                	je     402dac <__intel_cpu_features_init_body+0x3ec>
  402d9b:	48 89 e0             	mov    %rsp,%rax
  402d9e:	b9 1b 00 00 00       	mov    $0x1b,%ecx
  402da3:	e8 78 12 00 00       	call   404020 <__libirc_set_cpu_feature>
  402da8:	85 c0                	test   %eax,%eax
  402daa:	75 46                	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402dac:	f7 c7 00 00 80 00    	test   $0x800000,%edi
  402db2:	0f 85 3a 02 00 00    	jne    402ff2 <__intel_cpu_features_init_body+0x632>
  402db8:	41 f7 c0 00 00 00 08 	test   $0x8000000,%r8d
  402dbf:	0f 85 88 02 00 00    	jne    40304d <__intel_cpu_features_init_body+0x68d>
  402dc5:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
  402dca:	e8 b1 12 00 00       	call   404080 <__libirc_handle_intel_isa_disable>
  402dcf:	85 c0                	test   %eax,%eax
  402dd1:	0f 8e 09 06 00 00    	jle    4033e0 <__intel_cpu_features_init_body+0xa20>
  402dd7:	0f 28 44 24 20       	movaps 0x20(%rsp),%xmm0
  402ddc:	0f 55 04 24          	andnps (%rsp),%xmm0
  402de0:	e9 ff 05 00 00       	jmp    4033e4 <__intel_cpu_features_init_body+0xa24>
  402de5:	0f 28 04 24          	movaps (%rsp),%xmm0
  402de9:	0f 29 05 e0 52 00 00 	movaps %xmm0,0x52e0(%rip)        # 4080d0 <__intel_cpu_feature_indicator>
  402df0:	31 c0                	xor    %eax,%eax
  402df2:	0f 28 44 24 30       	movaps 0x30(%rsp),%xmm0
  402df7:	0f 28 4c 24 40       	movaps 0x40(%rsp),%xmm1
  402dfc:	0f 28 54 24 50       	movaps 0x50(%rsp),%xmm2
  402e01:	0f 28 5c 24 60       	movaps 0x60(%rsp),%xmm3
  402e06:	0f 28 64 24 70       	movaps 0x70(%rsp),%xmm4
  402e0b:	0f 28 ac 24 80 00 00 	movaps 0x80(%rsp),%xmm5
  402e12:	00 
  402e13:	0f 28 b4 24 90 00 00 	movaps 0x90(%rsp),%xmm6
  402e1a:	00 
  402e1b:	0f 28 bc 24 a0 00 00 	movaps 0xa0(%rsp),%xmm7
  402e22:	00 
  402e23:	44 0f 28 84 24 b0 00 	movaps 0xb0(%rsp),%xmm8
  402e2a:	00 00 
  402e2c:	44 0f 28 8c 24 c0 00 	movaps 0xc0(%rsp),%xmm9
  402e33:	00 00 
  402e35:	44 0f 28 94 24 d0 00 	movaps 0xd0(%rsp),%xmm10
  402e3c:	00 00 
  402e3e:	44 0f 28 9c 24 e0 00 	movaps 0xe0(%rsp),%xmm11
  402e45:	00 00 
  402e47:	44 0f 28 a4 24 f0 00 	movaps 0xf0(%rsp),%xmm12
  402e4e:	00 00 
  402e50:	44 0f 28 ac 24 00 01 	movaps 0x100(%rsp),%xmm13
  402e57:	00 00 
  402e59:	44 0f 28 b4 24 10 01 	movaps 0x110(%rsp),%xmm14
  402e60:	00 00 
  402e62:	44 0f 28 bc 24 20 01 	movaps 0x120(%rsp),%xmm15
  402e69:	00 00 
  402e6b:	48 81 c4 38 01 00 00 	add    $0x138,%rsp
  402e72:	5b                   	pop    %rbx
  402e73:	5d                   	pop    %rbp
  402e74:	5f                   	pop    %rdi
  402e75:	5e                   	pop    %rsi
  402e76:	59                   	pop    %rcx
  402e77:	5a                   	pop    %rdx
  402e78:	41 58                	pop    %r8
  402e7a:	41 59                	pop    %r9
  402e7c:	41 5a                	pop    %r10
  402e7e:	41 5b                	pop    %r11
  402e80:	c3                   	ret
  402e81:	48 89 e0             	mov    %rsp,%rax
  402e84:	b9 05 00 00 00       	mov    $0x5,%ecx
  402e89:	e8 92 11 00 00       	call   404020 <__libirc_set_cpu_feature>
  402e8e:	85 c0                	test   %eax,%eax
  402e90:	0f 85 5c ff ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402e96:	41 f7 c2 00 00 00 02 	test   $0x2000000,%r10d
  402e9d:	74 15                	je     402eb4 <__intel_cpu_features_init_body+0x4f4>
  402e9f:	48 89 e0             	mov    %rsp,%rax
  402ea2:	b9 06 00 00 00       	mov    $0x6,%ecx
  402ea7:	e8 74 11 00 00       	call   404020 <__libirc_set_cpu_feature>
  402eac:	85 c0                	test   %eax,%eax
  402eae:	0f 85 3e ff ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402eb4:	41 f7 c2 00 00 00 04 	test   $0x4000000,%r10d
  402ebb:	74 15                	je     402ed2 <__intel_cpu_features_init_body+0x512>
  402ebd:	48 89 e0             	mov    %rsp,%rax
  402ec0:	b9 07 00 00 00       	mov    $0x7,%ecx
  402ec5:	e8 56 11 00 00       	call   404020 <__libirc_set_cpu_feature>
  402eca:	85 c0                	test   %eax,%eax
  402ecc:	0f 85 20 ff ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402ed2:	41 f6 c0 01          	test   $0x1,%r8b
  402ed6:	74 15                	je     402eed <__intel_cpu_features_init_body+0x52d>
  402ed8:	48 89 e0             	mov    %rsp,%rax
  402edb:	b9 08 00 00 00       	mov    $0x8,%ecx
  402ee0:	e8 3b 11 00 00       	call   404020 <__libirc_set_cpu_feature>
  402ee5:	85 c0                	test   %eax,%eax
  402ee7:	0f 85 05 ff ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402eed:	41 f7 c0 00 02 00 00 	test   $0x200,%r8d
  402ef4:	74 15                	je     402f0b <__intel_cpu_features_init_body+0x54b>
  402ef6:	48 89 e0             	mov    %rsp,%rax
  402ef9:	b9 09 00 00 00       	mov    $0x9,%ecx
  402efe:	e8 1d 11 00 00       	call   404020 <__libirc_set_cpu_feature>
  402f03:	85 c0                	test   %eax,%eax
  402f05:	0f 85 e7 fe ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402f0b:	41 f7 c0 00 00 40 00 	test   $0x400000,%r8d
  402f12:	74 15                	je     402f29 <__intel_cpu_features_init_body+0x569>
  402f14:	48 89 e0             	mov    %rsp,%rax
  402f17:	b9 0c 00 00 00       	mov    $0xc,%ecx
  402f1c:	e8 ff 10 00 00       	call   404020 <__libirc_set_cpu_feature>
  402f21:	85 c0                	test   %eax,%eax
  402f23:	0f 85 c9 fe ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402f29:	41 f7 c0 00 00 08 00 	test   $0x80000,%r8d
  402f30:	74 15                	je     402f47 <__intel_cpu_features_init_body+0x587>
  402f32:	48 89 e0             	mov    %rsp,%rax
  402f35:	b9 0a 00 00 00       	mov    $0xa,%ecx
  402f3a:	e8 e1 10 00 00       	call   404020 <__libirc_set_cpu_feature>
  402f3f:	85 c0                	test   %eax,%eax
  402f41:	0f 85 ab fe ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402f47:	41 f7 c0 00 00 10 00 	test   $0x100000,%r8d
  402f4e:	74 15                	je     402f65 <__intel_cpu_features_init_body+0x5a5>
  402f50:	48 89 e0             	mov    %rsp,%rax
  402f53:	b9 0b 00 00 00       	mov    $0xb,%ecx
  402f58:	e8 c3 10 00 00       	call   404020 <__libirc_set_cpu_feature>
  402f5d:	85 c0                	test   %eax,%eax
  402f5f:	0f 85 8d fe ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402f65:	41 f7 c0 00 00 80 00 	test   $0x800000,%r8d
  402f6c:	74 15                	je     402f83 <__intel_cpu_features_init_body+0x5c3>
  402f6e:	48 89 e0             	mov    %rsp,%rax
  402f71:	b9 0d 00 00 00       	mov    $0xd,%ecx
  402f76:	e8 a5 10 00 00       	call   404020 <__libirc_set_cpu_feature>
  402f7b:	85 c0                	test   %eax,%eax
  402f7d:	0f 85 6f fe ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402f83:	41 f6 c0 02          	test   $0x2,%r8b
  402f87:	74 15                	je     402f9e <__intel_cpu_features_init_body+0x5de>
  402f89:	48 89 e0             	mov    %rsp,%rax
  402f8c:	b9 0e 00 00 00       	mov    $0xe,%ecx
  402f91:	e8 8a 10 00 00       	call   404020 <__libirc_set_cpu_feature>
  402f96:	85 c0                	test   %eax,%eax
  402f98:	0f 85 54 fe ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402f9e:	41 f7 c0 00 00 00 02 	test   $0x2000000,%r8d
  402fa5:	74 15                	je     402fbc <__intel_cpu_features_init_body+0x5fc>
  402fa7:	48 89 e0             	mov    %rsp,%rax
  402faa:	b9 0f 00 00 00       	mov    $0xf,%ecx
  402faf:	e8 6c 10 00 00       	call   404020 <__libirc_set_cpu_feature>
  402fb4:	85 c0                	test   %eax,%eax
  402fb6:	0f 85 36 fe ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402fbc:	b8 07 00 00 00       	mov    $0x7,%eax
  402fc1:	31 c9                	xor    %ecx,%ecx
  402fc3:	0f a2                	cpuid
  402fc5:	89 cf                	mov    %ecx,%edi
  402fc7:	89 d6                	mov    %edx,%esi
  402fc9:	41 89 d9             	mov    %ebx,%r9d
  402fcc:	f7 c3 00 00 00 20    	test   $0x20000000,%ebx
  402fd2:	0f 84 55 fb ff ff    	je     402b2d <__intel_cpu_features_init_body+0x16d>
  402fd8:	48 89 e0             	mov    %rsp,%rax
  402fdb:	b9 24 00 00 00       	mov    $0x24,%ecx
  402fe0:	e8 3b 10 00 00       	call   404020 <__libirc_set_cpu_feature>
  402fe5:	85 c0                	test   %eax,%eax
  402fe7:	0f 85 05 fe ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  402fed:	e9 3b fb ff ff       	jmp    402b2d <__intel_cpu_features_init_body+0x16d>
  402ff2:	48 89 e0             	mov    %rsp,%rax
  402ff5:	b9 01 00 00 00       	mov    $0x1,%ecx
  402ffa:	e8 21 10 00 00       	call   404020 <__libirc_set_cpu_feature>
  402fff:	85 c0                	test   %eax,%eax
  403001:	0f 85 eb fd ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403007:	b8 19 00 00 00       	mov    $0x19,%eax
  40300c:	31 c9                	xor    %ecx,%ecx
  40300e:	0f a2                	cpuid
  403010:	f6 c3 01             	test   $0x1,%bl
  403013:	74 15                	je     40302a <__intel_cpu_features_init_body+0x66a>
  403015:	48 89 e0             	mov    %rsp,%rax
  403018:	b9 45 00 00 00       	mov    $0x45,%ecx
  40301d:	e8 fe 0f 00 00       	call   404020 <__libirc_set_cpu_feature>
  403022:	85 c0                	test   %eax,%eax
  403024:	0f 85 c8 fd ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  40302a:	f6 c3 04             	test   $0x4,%bl
  40302d:	0f 84 85 fd ff ff    	je     402db8 <__intel_cpu_features_init_body+0x3f8>
  403033:	48 89 e0             	mov    %rsp,%rax
  403036:	b9 46 00 00 00       	mov    $0x46,%ecx
  40303b:	e8 e0 0f 00 00       	call   404020 <__libirc_set_cpu_feature>
  403040:	85 c0                	test   %eax,%eax
  403042:	0f 85 aa fd ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403048:	e9 6b fd ff ff       	jmp    402db8 <__intel_cpu_features_init_body+0x3f8>
  40304d:	48 89 e0             	mov    %rsp,%rax
  403050:	b9 01 00 00 00       	mov    $0x1,%ecx
  403055:	e8 c6 0f 00 00       	call   404020 <__libirc_set_cpu_feature>
  40305a:	85 c0                	test   %eax,%eax
  40305c:	0f 85 90 fd ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403062:	31 c9                	xor    %ecx,%ecx
  403064:	0f 01 d0             	xgetbv
  403067:	41 89 c2             	mov    %eax,%r10d
  40306a:	41 f7 d2             	not    %r10d
  40306d:	41 f7 c2 00 00 06 00 	test   $0x60000,%r10d
  403074:	75 6c                	jne    4030e2 <__intel_cpu_features_init_body+0x722>
  403076:	48 89 e0             	mov    %rsp,%rax
  403079:	b9 01 00 00 00       	mov    $0x1,%ecx
  40307e:	e8 9d 0f 00 00       	call   404020 <__libirc_set_cpu_feature>
  403083:	85 c0                	test   %eax,%eax
  403085:	0f 85 67 fd ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  40308b:	f7 c6 00 00 00 01    	test   $0x1000000,%esi
  403091:	74 15                	je     4030a8 <__intel_cpu_features_init_body+0x6e8>
  403093:	48 89 e0             	mov    %rsp,%rax
  403096:	b9 42 00 00 00       	mov    $0x42,%ecx
  40309b:	e8 80 0f 00 00       	call   404020 <__libirc_set_cpu_feature>
  4030a0:	85 c0                	test   %eax,%eax
  4030a2:	0f 85 4a fd ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  4030a8:	f7 c6 00 00 00 02    	test   $0x2000000,%esi
  4030ae:	74 15                	je     4030c5 <__intel_cpu_features_init_body+0x705>
  4030b0:	48 89 e0             	mov    %rsp,%rax
  4030b3:	b9 43 00 00 00       	mov    $0x43,%ecx
  4030b8:	e8 63 0f 00 00       	call   404020 <__libirc_set_cpu_feature>
  4030bd:	85 c0                	test   %eax,%eax
  4030bf:	0f 85 2d fd ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  4030c5:	f7 c6 00 00 40 00    	test   $0x400000,%esi
  4030cb:	74 15                	je     4030e2 <__intel_cpu_features_init_body+0x722>
  4030cd:	48 89 e0             	mov    %rsp,%rax
  4030d0:	b9 44 00 00 00       	mov    $0x44,%ecx
  4030d5:	e8 46 0f 00 00       	call   404020 <__libirc_set_cpu_feature>
  4030da:	85 c0                	test   %eax,%eax
  4030dc:	0f 85 10 fd ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  4030e2:	41 f6 c2 06          	test   $0x6,%r10b
  4030e6:	0f 85 d9 fc ff ff    	jne    402dc5 <__intel_cpu_features_init_body+0x405>
  4030ec:	48 89 e0             	mov    %rsp,%rax
  4030ef:	b9 01 00 00 00       	mov    $0x1,%ecx
  4030f4:	e8 27 0f 00 00       	call   404020 <__libirc_set_cpu_feature>
  4030f9:	85 c0                	test   %eax,%eax
  4030fb:	0f 85 f1 fc ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403101:	41 f7 c0 00 00 00 10 	test   $0x10000000,%r8d
  403108:	0f 85 f1 02 00 00    	jne    4033ff <__intel_cpu_features_init_body+0xa3f>
  40310e:	41 f7 c0 00 00 00 20 	test   $0x20000000,%r8d
  403115:	74 15                	je     40312c <__intel_cpu_features_init_body+0x76c>
  403117:	48 89 e0             	mov    %rsp,%rax
  40311a:	b9 11 00 00 00       	mov    $0x11,%ecx
  40311f:	e8 fc 0e 00 00       	call   404020 <__libirc_set_cpu_feature>
  403124:	85 c0                	test   %eax,%eax
  403126:	0f 85 c6 fc ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  40312c:	41 f6 c1 20          	test   $0x20,%r9b
  403130:	74 15                	je     403147 <__intel_cpu_features_init_body+0x787>
  403132:	48 89 e0             	mov    %rsp,%rax
  403135:	b9 18 00 00 00       	mov    $0x18,%ecx
  40313a:	e8 e1 0e 00 00       	call   404020 <__libirc_set_cpu_feature>
  40313f:	85 c0                	test   %eax,%eax
  403141:	0f 85 ab fc ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403147:	41 f7 c0 00 10 00 00 	test   $0x1000,%r8d
  40314e:	74 15                	je     403165 <__intel_cpu_features_init_body+0x7a5>
  403150:	48 89 e0             	mov    %rsp,%rax
  403153:	b9 13 00 00 00       	mov    $0x13,%ecx
  403158:	e8 c3 0e 00 00       	call   404020 <__libirc_set_cpu_feature>
  40315d:	85 c0                	test   %eax,%eax
  40315f:	0f 85 8d fc ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403165:	41 f6 c2 18          	test   $0x18,%r10b
  403169:	75 33                	jne    40319e <__intel_cpu_features_init_body+0x7de>
  40316b:	48 89 e0             	mov    %rsp,%rax
  40316e:	b9 01 00 00 00       	mov    $0x1,%ecx
  403173:	e8 a8 0e 00 00       	call   404020 <__libirc_set_cpu_feature>
  403178:	85 c0                	test   %eax,%eax
  40317a:	0f 85 72 fc ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403180:	41 f7 c1 00 40 00 00 	test   $0x4000,%r9d
  403187:	74 15                	je     40319e <__intel_cpu_features_init_body+0x7de>
  403189:	48 89 e0             	mov    %rsp,%rax
  40318c:	b9 25 00 00 00       	mov    $0x25,%ecx
  403191:	e8 8a 0e 00 00       	call   404020 <__libirc_set_cpu_feature>
  403196:	85 c0                	test   %eax,%eax
  403198:	0f 85 54 fc ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  40319e:	b8 07 00 00 00       	mov    $0x7,%eax
  4031a3:	b9 01 00 00 00       	mov    $0x1,%ecx
  4031a8:	0f a2                	cpuid
  4031aa:	89 c2                	mov    %eax,%edx
  4031ac:	f6 c2 10             	test   $0x10,%dl
  4031af:	74 15                	je     4031c6 <__intel_cpu_features_init_body+0x806>
  4031b1:	48 89 e0             	mov    %rsp,%rax
  4031b4:	b9 41 00 00 00       	mov    $0x41,%ecx
  4031b9:	e8 62 0e 00 00       	call   404020 <__libirc_set_cpu_feature>
  4031be:	85 c0                	test   %eax,%eax
  4031c0:	0f 85 2c fc ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  4031c6:	41 f6 c2 e0          	test   $0xe0,%r10b
  4031ca:	0f 85 f5 fb ff ff    	jne    402dc5 <__intel_cpu_features_init_body+0x405>
  4031d0:	48 89 e0             	mov    %rsp,%rax
  4031d3:	b9 01 00 00 00       	mov    $0x1,%ecx
  4031d8:	e8 43 0e 00 00       	call   404020 <__libirc_set_cpu_feature>
  4031dd:	85 c0                	test   %eax,%eax
  4031df:	0f 85 0d fc ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  4031e5:	41 f7 c1 00 00 01 00 	test   $0x10000,%r9d
  4031ec:	74 15                	je     403203 <__intel_cpu_features_init_body+0x843>
  4031ee:	48 89 e0             	mov    %rsp,%rax
  4031f1:	b9 19 00 00 00       	mov    $0x19,%ecx
  4031f6:	e8 25 0e 00 00       	call   404020 <__libirc_set_cpu_feature>
  4031fb:	85 c0                	test   %eax,%eax
  4031fd:	0f 85 ef fb ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403203:	41 f7 c1 00 00 00 10 	test   $0x10000000,%r9d
  40320a:	74 15                	je     403221 <__intel_cpu_features_init_body+0x861>
  40320c:	48 89 e0             	mov    %rsp,%rax
  40320f:	b9 23 00 00 00       	mov    $0x23,%ecx
  403214:	e8 07 0e 00 00       	call   404020 <__libirc_set_cpu_feature>
  403219:	85 c0                	test   %eax,%eax
  40321b:	0f 85 d1 fb ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403221:	41 f7 c1 00 00 00 08 	test   $0x8000000,%r9d
  403228:	74 15                	je     40323f <__intel_cpu_features_init_body+0x87f>
  40322a:	48 89 e0             	mov    %rsp,%rax
  40322d:	b9 21 00 00 00       	mov    $0x21,%ecx
  403232:	e8 e9 0d 00 00       	call   404020 <__libirc_set_cpu_feature>
  403237:	85 c0                	test   %eax,%eax
  403239:	0f 85 b3 fb ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  40323f:	41 f7 c1 00 00 00 04 	test   $0x4000000,%r9d
  403246:	74 15                	je     40325d <__intel_cpu_features_init_body+0x89d>
  403248:	48 89 e0             	mov    %rsp,%rax
  40324b:	b9 22 00 00 00       	mov    $0x22,%ecx
  403250:	e8 cb 0d 00 00       	call   404020 <__libirc_set_cpu_feature>
  403255:	85 c0                	test   %eax,%eax
  403257:	0f 85 95 fb ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  40325d:	41 f7 c1 00 00 02 00 	test   $0x20000,%r9d
  403264:	74 15                	je     40327b <__intel_cpu_features_init_body+0x8bb>
  403266:	48 89 e0             	mov    %rsp,%rax
  403269:	b9 1a 00 00 00       	mov    $0x1a,%ecx
  40326e:	e8 ad 0d 00 00       	call   404020 <__libirc_set_cpu_feature>
  403273:	85 c0                	test   %eax,%eax
  403275:	0f 85 77 fb ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  40327b:	41 f7 c1 00 00 00 40 	test   $0x40000000,%r9d
  403282:	74 15                	je     403299 <__intel_cpu_features_init_body+0x8d9>
  403284:	48 89 e0             	mov    %rsp,%rax
  403287:	b9 26 00 00 00       	mov    $0x26,%ecx
  40328c:	e8 8f 0d 00 00       	call   404020 <__libirc_set_cpu_feature>
  403291:	85 c0                	test   %eax,%eax
  403293:	0f 85 59 fb ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403299:	45 85 c9             	test   %r9d,%r9d
  40329c:	0f 88 b5 01 00 00    	js     403457 <__intel_cpu_features_init_body+0xa97>
  4032a2:	41 f7 c1 00 00 20 00 	test   $0x200000,%r9d
  4032a9:	74 15                	je     4032c0 <__intel_cpu_features_init_body+0x900>
  4032ab:	48 89 e0             	mov    %rsp,%rax
  4032ae:	b9 1f 00 00 00       	mov    $0x1f,%ecx
  4032b3:	e8 68 0d 00 00       	call   404020 <__libirc_set_cpu_feature>
  4032b8:	85 c0                	test   %eax,%eax
  4032ba:	0f 85 32 fb ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  4032c0:	40 f6 c7 02          	test   $0x2,%dil
  4032c4:	74 15                	je     4032db <__intel_cpu_features_init_body+0x91b>
  4032c6:	48 89 e0             	mov    %rsp,%rax
  4032c9:	b9 28 00 00 00       	mov    $0x28,%ecx
  4032ce:	e8 4d 0d 00 00       	call   404020 <__libirc_set_cpu_feature>
  4032d3:	85 c0                	test   %eax,%eax
  4032d5:	0f 85 17 fb ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  4032db:	f7 c7 00 40 00 00    	test   $0x4000,%edi
  4032e1:	74 15                	je     4032f8 <__intel_cpu_features_init_body+0x938>
  4032e3:	48 89 e0             	mov    %rsp,%rax
  4032e6:	b9 2b 00 00 00       	mov    $0x2b,%ecx
  4032eb:	e8 30 0d 00 00       	call   404020 <__libirc_set_cpu_feature>
  4032f0:	85 c0                	test   %eax,%eax
  4032f2:	0f 85 fa fa ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  4032f8:	40 f6 c6 04          	test   $0x4,%sil
  4032fc:	74 15                	je     403313 <__intel_cpu_features_init_body+0x953>
  4032fe:	48 89 e0             	mov    %rsp,%rax
  403301:	b9 2a 00 00 00       	mov    $0x2a,%ecx
  403306:	e8 15 0d 00 00       	call   404020 <__libirc_set_cpu_feature>
  40330b:	85 c0                	test   %eax,%eax
  40330d:	0f 85 df fa ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403313:	40 f6 c6 08          	test   $0x8,%sil
  403317:	74 15                	je     40332e <__intel_cpu_features_init_body+0x96e>
  403319:	48 89 e0             	mov    %rsp,%rax
  40331c:	b9 29 00 00 00       	mov    $0x29,%ecx
  403321:	e8 fa 0c 00 00       	call   404020 <__libirc_set_cpu_feature>
  403326:	85 c0                	test   %eax,%eax
  403328:	0f 85 c4 fa ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  40332e:	f7 c7 00 10 00 00    	test   $0x1000,%edi
  403334:	74 15                	je     40334b <__intel_cpu_features_init_body+0x98b>
  403336:	48 89 e0             	mov    %rsp,%rax
  403339:	b9 2c 00 00 00       	mov    $0x2c,%ecx
  40333e:	e8 dd 0c 00 00       	call   404020 <__libirc_set_cpu_feature>
  403343:	85 c0                	test   %eax,%eax
  403345:	0f 85 a7 fa ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  40334b:	40 f6 c7 40          	test   $0x40,%dil
  40334f:	74 15                	je     403366 <__intel_cpu_features_init_body+0x9a6>
  403351:	48 89 e0             	mov    %rsp,%rax
  403354:	b9 2d 00 00 00       	mov    $0x2d,%ecx
  403359:	e8 c2 0c 00 00       	call   404020 <__libirc_set_cpu_feature>
  40335e:	85 c0                	test   %eax,%eax
  403360:	0f 85 8c fa ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403366:	f7 c7 00 08 00 00    	test   $0x800,%edi
  40336c:	74 15                	je     403383 <__intel_cpu_features_init_body+0x9c3>
  40336e:	48 89 e0             	mov    %rsp,%rax
  403371:	b9 31 00 00 00       	mov    $0x31,%ecx
  403376:	e8 a5 0c 00 00       	call   404020 <__libirc_set_cpu_feature>
  40337b:	85 c0                	test   %eax,%eax
  40337d:	0f 85 6f fa ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403383:	f6 c2 20             	test   $0x20,%dl
  403386:	74 15                	je     40339d <__intel_cpu_features_init_body+0x9dd>
  403388:	48 89 e0             	mov    %rsp,%rax
  40338b:	b9 3f 00 00 00       	mov    $0x3f,%ecx
  403390:	e8 8b 0c 00 00       	call   404020 <__libirc_set_cpu_feature>
  403395:	85 c0                	test   %eax,%eax
  403397:	0f 85 55 fa ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  40339d:	f7 c6 00 00 80 00    	test   $0x800000,%esi
  4033a3:	74 15                	je     4033ba <__intel_cpu_features_init_body+0x9fa>
  4033a5:	48 89 e0             	mov    %rsp,%rax
  4033a8:	b9 3a 00 00 00       	mov    $0x3a,%ecx
  4033ad:	e8 6e 0c 00 00       	call   404020 <__libirc_set_cpu_feature>
  4033b2:	85 c0                	test   %eax,%eax
  4033b4:	0f 85 38 fa ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  4033ba:	f7 c6 00 01 00 00    	test   $0x100,%esi
  4033c0:	0f 84 ff f9 ff ff    	je     402dc5 <__intel_cpu_features_init_body+0x405>
  4033c6:	48 89 e0             	mov    %rsp,%rax
  4033c9:	b9 39 00 00 00       	mov    $0x39,%ecx
  4033ce:	e8 4d 0c 00 00       	call   404020 <__libirc_set_cpu_feature>
  4033d3:	85 c0                	test   %eax,%eax
  4033d5:	0f 85 17 fa ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  4033db:	e9 e5 f9 ff ff       	jmp    402dc5 <__intel_cpu_features_init_body+0x405>
  4033e0:	0f 28 04 24          	movaps (%rsp),%xmm0
  4033e4:	83 fd 01             	cmp    $0x1,%ebp
  4033e7:	75 07                	jne    4033f0 <__intel_cpu_features_init_body+0xa30>
  4033e9:	0f 29 05 e0 4c 00 00 	movaps %xmm0,0x4ce0(%rip)        # 4080d0 <__intel_cpu_feature_indicator>
  4033f0:	48 c7 c0 e0 80 40 00 	mov    $0x4080e0,%rax
  4033f7:	0f 29 00             	movaps %xmm0,(%rax)
  4033fa:	e9 f1 f9 ff ff       	jmp    402df0 <__intel_cpu_features_init_body+0x430>
  4033ff:	48 89 e0             	mov    %rsp,%rax
  403402:	b9 10 00 00 00       	mov    $0x10,%ecx
  403407:	e8 14 0c 00 00       	call   404020 <__libirc_set_cpu_feature>
  40340c:	85 c0                	test   %eax,%eax
  40340e:	0f 85 de f9 ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403414:	f7 c7 00 02 00 00    	test   $0x200,%edi
  40341a:	74 15                	je     403431 <__intel_cpu_features_init_body+0xa71>
  40341c:	48 89 e0             	mov    %rsp,%rax
  40341f:	b9 2f 00 00 00       	mov    $0x2f,%ecx
  403424:	e8 f7 0b 00 00       	call   404020 <__libirc_set_cpu_feature>
  403429:	85 c0                	test   %eax,%eax
  40342b:	0f 85 c1 f9 ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403431:	f7 c7 00 04 00 00    	test   $0x400,%edi
  403437:	0f 84 d1 fc ff ff    	je     40310e <__intel_cpu_features_init_body+0x74e>
  40343d:	48 89 e0             	mov    %rsp,%rax
  403440:	b9 30 00 00 00       	mov    $0x30,%ecx
  403445:	e8 d6 0b 00 00       	call   404020 <__libirc_set_cpu_feature>
  40344a:	85 c0                	test   %eax,%eax
  40344c:	0f 85 a0 f9 ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  403452:	e9 b7 fc ff ff       	jmp    40310e <__intel_cpu_features_init_body+0x74e>
  403457:	48 89 e0             	mov    %rsp,%rax
  40345a:	b9 27 00 00 00       	mov    $0x27,%ecx
  40345f:	e8 bc 0b 00 00       	call   404020 <__libirc_set_cpu_feature>
  403464:	85 c0                	test   %eax,%eax
  403466:	0f 85 86 f9 ff ff    	jne    402df2 <__intel_cpu_features_init_body+0x432>
  40346c:	e9 31 fe ff ff       	jmp    4032a2 <__intel_cpu_features_init_body+0x8e2>
  403471:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  403478:	00 00 00 
  40347b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000403480 <__intel_cpu_features_init_x>:
  403480:	f3 0f 1e fa          	endbr64
  403484:	50                   	push   %rax
  403485:	31 c0                	xor    %eax,%eax
  403487:	e8 34 f5 ff ff       	call   4029c0 <__intel_cpu_features_init_body>
  40348c:	48 83 c4 08          	add    $0x8,%rsp
  403490:	c3                   	ret
  403491:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  403498:	00 00 00 
  40349b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004034a0 <__libirc_get_feature_name>:
  4034a0:	f3 0f 1e fa          	endbr64
  4034a4:	50                   	push   %rax
  4034a5:	80 3d 44 4c 00 00 00 	cmpb   $0x0,0x4c44(%rip)        # 4080f0 <__libirc_isa_info_initialized>
  4034ac:	75 05                	jne    4034b3 <__libirc_get_feature_name+0x13>
  4034ae:	e8 1d 00 00 00       	call   4034d0 <__libirc_isa_init_once>
  4034b3:	89 f8                	mov    %edi,%eax
  4034b5:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  4034b9:	48 8d 0d 40 4c 00 00 	lea    0x4c40(%rip),%rcx        # 408100 <proc_info_features>
  4034c0:	48 8b 04 c1          	mov    (%rcx,%rax,8),%rax
  4034c4:	59                   	pop    %rcx
  4034c5:	c3                   	ret
  4034c6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4034cd:	00 00 00 

00000000004034d0 <__libirc_isa_init_once>:
  4034d0:	51                   	push   %rcx
  4034d1:	80 3d 18 4c 00 00 00 	cmpb   $0x0,0x4c18(%rip)        # 4080f0 <__libirc_isa_info_initialized>
  4034d8:	0f 85 aa 0a 00 00    	jne    403f88 <__libirc_isa_init_once+0xab8>
  4034de:	b8 c8 00 00 00       	mov    $0xc8,%eax
  4034e3:	48 8d 0d 16 4c 00 00 	lea    0x4c16(%rip),%rcx        # 408100 <proc_info_features>
  4034ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4034f0:	c7 84 08 58 ff ff ff 	movl   $0xffffffff,-0xa8(%rax,%rcx,1)
  4034f7:	ff ff ff ff 
  4034fb:	c7 84 08 70 ff ff ff 	movl   $0xffffffff,-0x90(%rax,%rcx,1)
  403502:	ff ff ff ff 
  403506:	c7 44 08 88 ff ff ff 	movl   $0xffffffff,-0x78(%rax,%rcx,1)
  40350d:	ff 
  40350e:	c7 44 08 a0 ff ff ff 	movl   $0xffffffff,-0x60(%rax,%rcx,1)
  403515:	ff 
  403516:	c7 44 08 b8 ff ff ff 	movl   $0xffffffff,-0x48(%rax,%rcx,1)
  40351d:	ff 
  40351e:	c7 44 08 d0 ff ff ff 	movl   $0xffffffff,-0x30(%rax,%rcx,1)
  403525:	ff 
  403526:	c7 44 08 e8 ff ff ff 	movl   $0xffffffff,-0x18(%rax,%rcx,1)
  40352d:	ff 
  40352e:	c7 04 08 ff ff ff ff 	movl   $0xffffffff,(%rax,%rcx,1)
  403535:	48 05 c0 00 00 00    	add    $0xc0,%rax
  40353b:	48 3d c8 06 00 00    	cmp    $0x6c8,%rax
  403541:	75 ad                	jne    4034f0 <__libirc_isa_init_once+0x20>
  403543:	c7 05 d3 51 00 00 ff 	movl   $0xffffffff,0x51d3(%rip)        # 408720 <proc_info_features+0x620>
  40354a:	ff ff ff 
  40354d:	c7 05 e1 51 00 00 ff 	movl   $0xffffffff,0x51e1(%rip)        # 408738 <proc_info_features+0x638>
  403554:	ff ff ff 
  403557:	c7 05 ef 51 00 00 ff 	movl   $0xffffffff,0x51ef(%rip)        # 408750 <proc_info_features+0x650>
  40355e:	ff ff ff 
  403561:	c7 05 fd 51 00 00 ff 	movl   $0xffffffff,0x51fd(%rip)        # 408768 <proc_info_features+0x668>
  403568:	ff ff ff 
  40356b:	c7 05 0b 52 00 00 ff 	movl   $0xffffffff,0x520b(%rip)        # 408780 <proc_info_features+0x680>
  403572:	ff ff ff 
  403575:	c7 05 19 52 00 00 ff 	movl   $0xffffffff,0x5219(%rip)        # 408798 <proc_info_features+0x698>
  40357c:	ff ff ff 
  40357f:	48 8d 05 53 1b 00 00 	lea    0x1b53(%rip),%rax        # 4050d9 <_IO_stdin_used+0xd9>
  403586:	48 89 05 8b 4b 00 00 	mov    %rax,0x4b8b(%rip)        # 408118 <proc_info_features+0x18>
  40358d:	c7 05 89 4b 00 00 00 	movl   $0x0,0x4b89(%rip)        # 408120 <proc_info_features+0x20>
  403594:	00 00 00 
  403597:	48 8d 05 48 1b 00 00 	lea    0x1b48(%rip),%rax        # 4050e6 <_IO_stdin_used+0xe6>
  40359e:	48 89 05 8b 4b 00 00 	mov    %rax,0x4b8b(%rip)        # 408130 <proc_info_features+0x30>
  4035a5:	c7 05 89 4b 00 00 01 	movl   $0x1,0x4b89(%rip)        # 408138 <proc_info_features+0x38>
  4035ac:	00 00 00 
  4035af:	48 8d 05 34 1b 00 00 	lea    0x1b34(%rip),%rax        # 4050ea <_IO_stdin_used+0xea>
  4035b6:	48 89 05 8b 4b 00 00 	mov    %rax,0x4b8b(%rip)        # 408148 <proc_info_features+0x48>
  4035bd:	c7 05 89 4b 00 00 02 	movl   $0x2,0x4b89(%rip)        # 408150 <proc_info_features+0x50>
  4035c4:	00 00 00 
  4035c7:	c7 05 97 4b 00 00 03 	movl   $0x3,0x4b97(%rip)        # 408168 <proc_info_features+0x68>
  4035ce:	00 00 00 
  4035d1:	48 8d 05 17 1b 00 00 	lea    0x1b17(%rip),%rax        # 4050ef <_IO_stdin_used+0xef>
  4035d8:	48 89 05 91 4b 00 00 	mov    %rax,0x4b91(%rip)        # 408170 <proc_info_features+0x70>
  4035df:	48 8d 05 0d 1b 00 00 	lea    0x1b0d(%rip),%rax        # 4050f3 <_IO_stdin_used+0xf3>
  4035e6:	48 89 05 73 4b 00 00 	mov    %rax,0x4b73(%rip)        # 408160 <proc_info_features+0x60>
  4035ed:	48 8d 05 03 1b 00 00 	lea    0x1b03(%rip),%rax        # 4050f7 <_IO_stdin_used+0xf7>
  4035f4:	48 89 05 7d 4b 00 00 	mov    %rax,0x4b7d(%rip)        # 408178 <proc_info_features+0x78>
  4035fb:	c7 05 7b 4b 00 00 04 	movl   $0x4,0x4b7b(%rip)        # 408180 <proc_info_features+0x80>
  403602:	00 00 00 
  403605:	c7 05 89 4b 00 00 05 	movl   $0x5,0x4b89(%rip)        # 408198 <proc_info_features+0x98>
  40360c:	00 00 00 
  40360f:	48 8d 05 e8 1a 00 00 	lea    0x1ae8(%rip),%rax        # 4050fe <_IO_stdin_used+0xfe>
  403616:	48 89 05 83 4b 00 00 	mov    %rax,0x4b83(%rip)        # 4081a0 <proc_info_features+0xa0>
  40361d:	48 8d 05 de 1a 00 00 	lea    0x1ade(%rip),%rax        # 405102 <_IO_stdin_used+0x102>
  403624:	48 89 05 65 4b 00 00 	mov    %rax,0x4b65(%rip)        # 408190 <proc_info_features+0x90>
  40362b:	c7 05 7b 4b 00 00 06 	movl   $0x6,0x4b7b(%rip)        # 4081b0 <proc_info_features+0xb0>
  403632:	00 00 00 
  403635:	48 8d 05 ca 1a 00 00 	lea    0x1aca(%rip),%rax        # 405106 <_IO_stdin_used+0x106>
  40363c:	48 89 05 75 4b 00 00 	mov    %rax,0x4b75(%rip)        # 4081b8 <proc_info_features+0xb8>
  403643:	48 8d 05 c1 1a 00 00 	lea    0x1ac1(%rip),%rax        # 40510b <_IO_stdin_used+0x10b>
  40364a:	48 89 05 57 4b 00 00 	mov    %rax,0x4b57(%rip)        # 4081a8 <proc_info_features+0xa8>
  403651:	c7 05 6d 4b 00 00 07 	movl   $0x7,0x4b6d(%rip)        # 4081c8 <proc_info_features+0xc8>
  403658:	00 00 00 
  40365b:	48 8d 05 af 1a 00 00 	lea    0x1aaf(%rip),%rax        # 405111 <_IO_stdin_used+0x111>
  403662:	48 89 05 67 4b 00 00 	mov    %rax,0x4b67(%rip)        # 4081d0 <proc_info_features+0xd0>
  403669:	48 8d 05 a7 1a 00 00 	lea    0x1aa7(%rip),%rax        # 405117 <_IO_stdin_used+0x117>
  403670:	48 89 05 49 4b 00 00 	mov    %rax,0x4b49(%rip)        # 4081c0 <proc_info_features+0xc0>
  403677:	c7 05 5f 4b 00 00 08 	movl   $0x8,0x4b5f(%rip)        # 4081e0 <proc_info_features+0xe0>
  40367e:	00 00 00 
  403681:	48 8d 05 88 1a 00 00 	lea    0x1a88(%rip),%rax        # 405110 <_IO_stdin_used+0x110>
  403688:	48 89 05 59 4b 00 00 	mov    %rax,0x4b59(%rip)        # 4081e8 <proc_info_features+0xe8>
  40368f:	48 8d 05 80 1a 00 00 	lea    0x1a80(%rip),%rax        # 405116 <_IO_stdin_used+0x116>
  403696:	48 89 05 3b 4b 00 00 	mov    %rax,0x4b3b(%rip)        # 4081d8 <proc_info_features+0xd8>
  40369d:	c7 05 51 4b 00 00 09 	movl   $0x9,0x4b51(%rip)        # 4081f8 <proc_info_features+0xf8>
  4036a4:	00 00 00 
  4036a7:	48 8d 05 6e 1a 00 00 	lea    0x1a6e(%rip),%rax        # 40511c <_IO_stdin_used+0x11c>
  4036ae:	48 89 05 4b 4b 00 00 	mov    %rax,0x4b4b(%rip)        # 408200 <proc_info_features+0x100>
  4036b5:	48 8d 05 67 1a 00 00 	lea    0x1a67(%rip),%rax        # 405123 <_IO_stdin_used+0x123>
  4036bc:	48 89 05 2d 4b 00 00 	mov    %rax,0x4b2d(%rip)        # 4081f0 <proc_info_features+0xf0>
  4036c3:	c7 05 43 4b 00 00 0a 	movl   $0xa,0x4b43(%rip)        # 408210 <proc_info_features+0x110>
  4036ca:	00 00 00 
  4036cd:	48 8d 05 56 1a 00 00 	lea    0x1a56(%rip),%rax        # 40512a <_IO_stdin_used+0x12a>
  4036d4:	48 89 05 3d 4b 00 00 	mov    %rax,0x4b3d(%rip)        # 408218 <proc_info_features+0x118>
  4036db:	48 8d 05 4d 1a 00 00 	lea    0x1a4d(%rip),%rax        # 40512f <_IO_stdin_used+0x12f>
  4036e2:	48 89 05 1f 4b 00 00 	mov    %rax,0x4b1f(%rip)        # 408208 <proc_info_features+0x108>
  4036e9:	c7 05 35 4b 00 00 0b 	movl   $0xb,0x4b35(%rip)        # 408228 <proc_info_features+0x128>
  4036f0:	00 00 00 
  4036f3:	48 8d 05 3c 1a 00 00 	lea    0x1a3c(%rip),%rax        # 405136 <_IO_stdin_used+0x136>
  4036fa:	48 89 05 2f 4b 00 00 	mov    %rax,0x4b2f(%rip)        # 408230 <proc_info_features+0x130>
  403701:	48 8d 05 34 1a 00 00 	lea    0x1a34(%rip),%rax        # 40513c <_IO_stdin_used+0x13c>
  403708:	48 89 05 11 4b 00 00 	mov    %rax,0x4b11(%rip)        # 408220 <proc_info_features+0x120>
  40370f:	c7 05 27 4b 00 00 0c 	movl   $0xc,0x4b27(%rip)        # 408240 <proc_info_features+0x140>
  403716:	00 00 00 
  403719:	48 8d 05 22 1a 00 00 	lea    0x1a22(%rip),%rax        # 405142 <_IO_stdin_used+0x142>
  403720:	48 89 05 21 4b 00 00 	mov    %rax,0x4b21(%rip)        # 408248 <proc_info_features+0x148>
  403727:	48 8d 05 1b 1a 00 00 	lea    0x1a1b(%rip),%rax        # 405149 <_IO_stdin_used+0x149>
  40372e:	48 89 05 03 4b 00 00 	mov    %rax,0x4b03(%rip)        # 408238 <proc_info_features+0x138>
  403735:	c7 05 19 4b 00 00 0d 	movl   $0xd,0x4b19(%rip)        # 408258 <proc_info_features+0x158>
  40373c:	00 00 00 
  40373f:	48 8d 05 0a 1a 00 00 	lea    0x1a0a(%rip),%rax        # 405150 <_IO_stdin_used+0x150>
  403746:	48 89 05 13 4b 00 00 	mov    %rax,0x4b13(%rip)        # 408260 <proc_info_features+0x160>
  40374d:	48 8d 05 dc 1b 00 00 	lea    0x1bdc(%rip),%rax        # 405330 <_IO_stdin_used+0x330>
  403754:	48 89 05 f5 4a 00 00 	mov    %rax,0x4af5(%rip)        # 408250 <proc_info_features+0x150>
  40375b:	c7 05 0b 4b 00 00 0e 	movl   $0xe,0x4b0b(%rip)        # 408270 <proc_info_features+0x170>
  403762:	00 00 00 
  403765:	48 8d 05 af 1b 00 00 	lea    0x1baf(%rip),%rax        # 40531b <_IO_stdin_used+0x31b>
  40376c:	48 89 05 05 4b 00 00 	mov    %rax,0x4b05(%rip)        # 408278 <proc_info_features+0x178>
  403773:	48 8d 05 a6 1b 00 00 	lea    0x1ba6(%rip),%rax        # 405320 <_IO_stdin_used+0x320>
  40377a:	48 89 05 e7 4a 00 00 	mov    %rax,0x4ae7(%rip)        # 408268 <proc_info_features+0x168>
  403781:	c7 05 fd 4a 00 00 10 	movl   $0x10,0x4afd(%rip)        # 408288 <proc_info_features+0x188>
  403788:	00 00 00 
  40378b:	48 8d 05 c5 19 00 00 	lea    0x19c5(%rip),%rax        # 405157 <_IO_stdin_used+0x157>
  403792:	48 89 05 f7 4a 00 00 	mov    %rax,0x4af7(%rip)        # 408290 <proc_info_features+0x190>
  403799:	48 8d 05 bb 19 00 00 	lea    0x19bb(%rip),%rax        # 40515b <_IO_stdin_used+0x15b>
  4037a0:	48 89 05 d9 4a 00 00 	mov    %rax,0x4ad9(%rip)        # 408280 <proc_info_features+0x180>
  4037a7:	c7 05 ef 4a 00 00 0f 	movl   $0xf,0x4aef(%rip)        # 4082a0 <proc_info_features+0x1a0>
  4037ae:	00 00 00 
  4037b1:	48 8d 05 a7 19 00 00 	lea    0x19a7(%rip),%rax        # 40515f <_IO_stdin_used+0x15f>
  4037b8:	48 89 05 e9 4a 00 00 	mov    %rax,0x4ae9(%rip)        # 4082a8 <proc_info_features+0x1a8>
  4037bf:	48 8d 05 9e 19 00 00 	lea    0x199e(%rip),%rax        # 405164 <_IO_stdin_used+0x164>
  4037c6:	48 89 05 cb 4a 00 00 	mov    %rax,0x4acb(%rip)        # 408298 <proc_info_features+0x198>
  4037cd:	c7 05 e1 4a 00 00 11 	movl   $0x11,0x4ae1(%rip)        # 4082b8 <proc_info_features+0x1b8>
  4037d4:	00 00 00 
  4037d7:	48 8d 05 8b 19 00 00 	lea    0x198b(%rip),%rax        # 405169 <_IO_stdin_used+0x169>
  4037de:	48 89 05 db 4a 00 00 	mov    %rax,0x4adb(%rip)        # 4082c0 <proc_info_features+0x1c0>
  4037e5:	48 8d 05 83 19 00 00 	lea    0x1983(%rip),%rax        # 40516f <_IO_stdin_used+0x16f>
  4037ec:	48 89 05 bd 4a 00 00 	mov    %rax,0x4abd(%rip)        # 4082b0 <proc_info_features+0x1b0>
  4037f3:	c7 05 d3 4a 00 00 12 	movl   $0x12,0x4ad3(%rip)        # 4082d0 <proc_info_features+0x1d0>
  4037fa:	00 00 00 
  4037fd:	48 8d 05 f0 19 00 00 	lea    0x19f0(%rip),%rax        # 4051f4 <_IO_stdin_used+0x1f4>
  403804:	48 89 05 cd 4a 00 00 	mov    %rax,0x4acd(%rip)        # 4082d8 <proc_info_features+0x1d8>
  40380b:	48 8d 05 63 19 00 00 	lea    0x1963(%rip),%rax        # 405175 <_IO_stdin_used+0x175>
  403812:	48 89 05 af 4a 00 00 	mov    %rax,0x4aaf(%rip)        # 4082c8 <proc_info_features+0x1c8>
  403819:	c7 05 c5 4a 00 00 13 	movl   $0x13,0x4ac5(%rip)        # 4082e8 <proc_info_features+0x1e8>
  403820:	00 00 00 
  403823:	48 8d 05 4c 1a 00 00 	lea    0x1a4c(%rip),%rax        # 405276 <_IO_stdin_used+0x276>
  40382a:	48 89 05 bf 4a 00 00 	mov    %rax,0x4abf(%rip)        # 4082f0 <proc_info_features+0x1f0>
  403831:	48 8d 05 49 1a 00 00 	lea    0x1a49(%rip),%rax        # 405281 <_IO_stdin_used+0x281>
  403838:	48 89 05 a1 4a 00 00 	mov    %rax,0x4aa1(%rip)        # 4082e0 <proc_info_features+0x1e0>
  40383f:	c7 05 b7 4a 00 00 14 	movl   $0x14,0x4ab7(%rip)        # 408300 <proc_info_features+0x200>
  403846:	00 00 00 
  403849:	48 8d 05 29 19 00 00 	lea    0x1929(%rip),%rax        # 405179 <_IO_stdin_used+0x179>
  403850:	48 89 05 b1 4a 00 00 	mov    %rax,0x4ab1(%rip)        # 408308 <proc_info_features+0x208>
  403857:	48 8d 05 21 19 00 00 	lea    0x1921(%rip),%rax        # 40517f <_IO_stdin_used+0x17f>
  40385e:	48 89 05 93 4a 00 00 	mov    %rax,0x4a93(%rip)        # 4082f8 <proc_info_features+0x1f8>
  403865:	c7 05 a9 4a 00 00 15 	movl   $0x15,0x4aa9(%rip)        # 408318 <proc_info_features+0x218>
  40386c:	00 00 00 
  40386f:	48 8d 05 0f 19 00 00 	lea    0x190f(%rip),%rax        # 405185 <_IO_stdin_used+0x185>
  403876:	48 89 05 a3 4a 00 00 	mov    %rax,0x4aa3(%rip)        # 408320 <proc_info_features+0x220>
  40387d:	48 8d 05 05 19 00 00 	lea    0x1905(%rip),%rax        # 405189 <_IO_stdin_used+0x189>
  403884:	48 89 05 85 4a 00 00 	mov    %rax,0x4a85(%rip)        # 408310 <proc_info_features+0x210>
  40388b:	c7 05 9b 4a 00 00 16 	movl   $0x16,0x4a9b(%rip)        # 408330 <proc_info_features+0x230>
  403892:	00 00 00 
  403895:	48 8d 05 f1 18 00 00 	lea    0x18f1(%rip),%rax        # 40518d <_IO_stdin_used+0x18d>
  40389c:	48 89 05 95 4a 00 00 	mov    %rax,0x4a95(%rip)        # 408338 <proc_info_features+0x238>
  4038a3:	48 8d 05 e7 18 00 00 	lea    0x18e7(%rip),%rax        # 405191 <_IO_stdin_used+0x191>
  4038aa:	48 89 05 77 4a 00 00 	mov    %rax,0x4a77(%rip)        # 408328 <proc_info_features+0x228>
  4038b1:	c7 05 8d 4a 00 00 17 	movl   $0x17,0x4a8d(%rip)        # 408348 <proc_info_features+0x248>
  4038b8:	00 00 00 
  4038bb:	48 8d 05 d3 18 00 00 	lea    0x18d3(%rip),%rax        # 405195 <_IO_stdin_used+0x195>
  4038c2:	48 89 05 87 4a 00 00 	mov    %rax,0x4a87(%rip)        # 408350 <proc_info_features+0x250>
  4038c9:	48 8d 05 ca 18 00 00 	lea    0x18ca(%rip),%rax        # 40519a <_IO_stdin_used+0x19a>
  4038d0:	48 89 05 69 4a 00 00 	mov    %rax,0x4a69(%rip)        # 408340 <proc_info_features+0x240>
  4038d7:	c7 05 7f 4a 00 00 1b 	movl   $0x1b,0x4a7f(%rip)        # 408360 <proc_info_features+0x260>
  4038de:	00 00 00 
  4038e1:	48 8d 05 b7 18 00 00 	lea    0x18b7(%rip),%rax        # 40519f <_IO_stdin_used+0x19f>
  4038e8:	48 89 05 79 4a 00 00 	mov    %rax,0x4a79(%rip)        # 408368 <proc_info_features+0x268>
  4038ef:	48 8d 05 b1 18 00 00 	lea    0x18b1(%rip),%rax        # 4051a7 <_IO_stdin_used+0x1a7>
  4038f6:	48 89 05 5b 4a 00 00 	mov    %rax,0x4a5b(%rip)        # 408358 <proc_info_features+0x258>
  4038fd:	c7 05 71 4a 00 00 18 	movl   $0x18,0x4a71(%rip)        # 408378 <proc_info_features+0x278>
  403904:	00 00 00 
  403907:	48 8d 05 a1 18 00 00 	lea    0x18a1(%rip),%rax        # 4051af <_IO_stdin_used+0x1af>
  40390e:	48 89 05 6b 4a 00 00 	mov    %rax,0x4a6b(%rip)        # 408380 <proc_info_features+0x280>
  403915:	48 8d 05 9c 18 00 00 	lea    0x189c(%rip),%rax        # 4051b8 <_IO_stdin_used+0x1b8>
  40391c:	48 89 05 4d 4a 00 00 	mov    %rax,0x4a4d(%rip)        # 408370 <proc_info_features+0x270>
  403923:	c7 05 63 4a 00 00 19 	movl   $0x19,0x4a63(%rip)        # 408390 <proc_info_features+0x290>
  40392a:	00 00 00 
  40392d:	48 8d 05 8d 18 00 00 	lea    0x188d(%rip),%rax        # 4051c1 <_IO_stdin_used+0x1c1>
  403934:	48 89 05 5d 4a 00 00 	mov    %rax,0x4a5d(%rip)        # 408398 <proc_info_features+0x298>
  40393b:	48 8d 05 87 18 00 00 	lea    0x1887(%rip),%rax        # 4051c9 <_IO_stdin_used+0x1c9>
  403942:	48 89 05 3f 4a 00 00 	mov    %rax,0x4a3f(%rip)        # 408388 <proc_info_features+0x288>
  403949:	48 8d 05 81 18 00 00 	lea    0x1881(%rip),%rax        # 4051d1 <_IO_stdin_used+0x1d1>
  403950:	48 89 05 49 4a 00 00 	mov    %rax,0x4a49(%rip)        # 4083a0 <proc_info_features+0x2a0>
  403957:	c7 05 47 4a 00 00 1a 	movl   $0x1a,0x4a47(%rip)        # 4083a8 <proc_info_features+0x2a8>
  40395e:	00 00 00 
  403961:	c7 05 55 4a 00 00 1c 	movl   $0x1c,0x4a55(%rip)        # 4083c0 <proc_info_features+0x2c0>
  403968:	00 00 00 
  40396b:	48 8d 05 65 18 00 00 	lea    0x1865(%rip),%rax        # 4051d7 <_IO_stdin_used+0x1d7>
  403972:	48 89 05 4f 4a 00 00 	mov    %rax,0x4a4f(%rip)        # 4083c8 <proc_info_features+0x2c8>
  403979:	48 8d 05 5b 18 00 00 	lea    0x185b(%rip),%rax        # 4051db <_IO_stdin_used+0x1db>
  403980:	48 89 05 31 4a 00 00 	mov    %rax,0x4a31(%rip)        # 4083b8 <proc_info_features+0x2b8>
  403987:	c7 05 47 4a 00 00 1d 	movl   $0x1d,0x4a47(%rip)        # 4083d8 <proc_info_features+0x2d8>
  40398e:	00 00 00 
  403991:	48 8d 05 47 18 00 00 	lea    0x1847(%rip),%rax        # 4051df <_IO_stdin_used+0x1df>
  403998:	48 89 05 41 4a 00 00 	mov    %rax,0x4a41(%rip)        # 4083e0 <proc_info_features+0x2e0>
  40399f:	48 8d 05 40 18 00 00 	lea    0x1840(%rip),%rax        # 4051e6 <_IO_stdin_used+0x1e6>
  4039a6:	48 89 05 23 4a 00 00 	mov    %rax,0x4a23(%rip)        # 4083d0 <proc_info_features+0x2d0>
  4039ad:	c7 05 39 4a 00 00 1e 	movl   $0x1e,0x4a39(%rip)        # 4083f0 <proc_info_features+0x2f0>
  4039b4:	00 00 00 
  4039b7:	48 8d 05 2f 18 00 00 	lea    0x182f(%rip),%rax        # 4051ed <_IO_stdin_used+0x1ed>
  4039be:	48 89 05 33 4a 00 00 	mov    %rax,0x4a33(%rip)        # 4083f8 <proc_info_features+0x2f8>
  4039c5:	48 8d 05 2c 18 00 00 	lea    0x182c(%rip),%rax        # 4051f8 <_IO_stdin_used+0x1f8>
  4039cc:	48 89 05 15 4a 00 00 	mov    %rax,0x4a15(%rip)        # 4083e8 <proc_info_features+0x2e8>
  4039d3:	c7 05 2b 4a 00 00 ff 	movl   $0xffffffff,0x4a2b(%rip)        # 408408 <proc_info_features+0x308>
  4039da:	ff ff ff 
  4039dd:	c7 05 39 4a 00 00 20 	movl   $0x20,0x4a39(%rip)        # 408420 <proc_info_features+0x320>
  4039e4:	00 00 00 
  4039e7:	48 8d 05 17 18 00 00 	lea    0x1817(%rip),%rax        # 405205 <_IO_stdin_used+0x205>
  4039ee:	48 89 05 33 4a 00 00 	mov    %rax,0x4a33(%rip)        # 408428 <proc_info_features+0x328>
  4039f5:	48 8d 05 12 18 00 00 	lea    0x1812(%rip),%rax        # 40520e <_IO_stdin_used+0x20e>
  4039fc:	48 89 05 15 4a 00 00 	mov    %rax,0x4a15(%rip)        # 408418 <proc_info_features+0x318>
  403a03:	c7 05 2b 4a 00 00 21 	movl   $0x21,0x4a2b(%rip)        # 408438 <proc_info_features+0x338>
  403a0a:	00 00 00 
  403a0d:	48 8d 05 03 18 00 00 	lea    0x1803(%rip),%rax        # 405217 <_IO_stdin_used+0x217>
  403a14:	48 89 05 25 4a 00 00 	mov    %rax,0x4a25(%rip)        # 408440 <proc_info_features+0x340>
  403a1b:	48 8d 05 fe 17 00 00 	lea    0x17fe(%rip),%rax        # 405220 <_IO_stdin_used+0x220>
  403a22:	48 89 05 07 4a 00 00 	mov    %rax,0x4a07(%rip)        # 408430 <proc_info_features+0x330>
  403a29:	c7 05 1d 4a 00 00 22 	movl   $0x22,0x4a1d(%rip)        # 408450 <proc_info_features+0x350>
  403a30:	00 00 00 
  403a33:	48 8d 05 ef 17 00 00 	lea    0x17ef(%rip),%rax        # 405229 <_IO_stdin_used+0x229>
  403a3a:	48 89 05 17 4a 00 00 	mov    %rax,0x4a17(%rip)        # 408458 <proc_info_features+0x358>
  403a41:	48 8d 05 ea 17 00 00 	lea    0x17ea(%rip),%rax        # 405232 <_IO_stdin_used+0x232>
  403a48:	48 89 05 f9 49 00 00 	mov    %rax,0x49f9(%rip)        # 408448 <proc_info_features+0x348>
  403a4f:	c7 05 0f 4a 00 00 23 	movl   $0x23,0x4a0f(%rip)        # 408468 <proc_info_features+0x368>
  403a56:	00 00 00 
  403a59:	48 8d 05 db 17 00 00 	lea    0x17db(%rip),%rax        # 40523b <_IO_stdin_used+0x23b>
  403a60:	48 89 05 09 4a 00 00 	mov    %rax,0x4a09(%rip)        # 408470 <proc_info_features+0x370>
  403a67:	48 8d 05 d1 17 00 00 	lea    0x17d1(%rip),%rax        # 40523f <_IO_stdin_used+0x23f>
  403a6e:	48 89 05 eb 49 00 00 	mov    %rax,0x49eb(%rip)        # 408460 <proc_info_features+0x360>
  403a75:	c7 05 01 4a 00 00 24 	movl   $0x24,0x4a01(%rip)        # 408480 <proc_info_features+0x380>
  403a7c:	00 00 00 
  403a7f:	48 8d 05 bd 17 00 00 	lea    0x17bd(%rip),%rax        # 405243 <_IO_stdin_used+0x243>
  403a86:	48 89 05 fb 49 00 00 	mov    %rax,0x49fb(%rip)        # 408488 <proc_info_features+0x388>
  403a8d:	48 8d 05 b3 17 00 00 	lea    0x17b3(%rip),%rax        # 405247 <_IO_stdin_used+0x247>
  403a94:	48 89 05 dd 49 00 00 	mov    %rax,0x49dd(%rip)        # 408478 <proc_info_features+0x378>
  403a9b:	c7 05 f3 49 00 00 25 	movl   $0x25,0x49f3(%rip)        # 408498 <proc_info_features+0x398>
  403aa2:	00 00 00 
  403aa5:	48 8d 05 9f 17 00 00 	lea    0x179f(%rip),%rax        # 40524b <_IO_stdin_used+0x24b>
  403aac:	48 89 05 ed 49 00 00 	mov    %rax,0x49ed(%rip)        # 4084a0 <proc_info_features+0x3a0>
  403ab3:	48 8d 05 9a 17 00 00 	lea    0x179a(%rip),%rax        # 405254 <_IO_stdin_used+0x254>
  403aba:	48 89 05 cf 49 00 00 	mov    %rax,0x49cf(%rip)        # 408490 <proc_info_features+0x390>
  403ac1:	c7 05 e5 49 00 00 26 	movl   $0x26,0x49e5(%rip)        # 4084b0 <proc_info_features+0x3b0>
  403ac8:	00 00 00 
  403acb:	48 8d 05 8b 17 00 00 	lea    0x178b(%rip),%rax        # 40525d <_IO_stdin_used+0x25d>
  403ad2:	48 89 05 df 49 00 00 	mov    %rax,0x49df(%rip)        # 4084b8 <proc_info_features+0x3b8>
  403ad9:	48 8d 05 86 17 00 00 	lea    0x1786(%rip),%rax        # 405266 <_IO_stdin_used+0x266>
  403ae0:	48 89 05 c1 49 00 00 	mov    %rax,0x49c1(%rip)        # 4084a8 <proc_info_features+0x3a8>
  403ae7:	c7 05 d7 49 00 00 27 	movl   $0x27,0x49d7(%rip)        # 4084c8 <proc_info_features+0x3c8>
  403aee:	00 00 00 
  403af1:	48 8d 05 77 17 00 00 	lea    0x1777(%rip),%rax        # 40526f <_IO_stdin_used+0x26f>
  403af8:	48 89 05 d1 49 00 00 	mov    %rax,0x49d1(%rip)        # 4084d0 <proc_info_features+0x3d0>
  403aff:	48 8d 05 74 17 00 00 	lea    0x1774(%rip),%rax        # 40527a <_IO_stdin_used+0x27a>
  403b06:	48 89 05 b3 49 00 00 	mov    %rax,0x49b3(%rip)        # 4084c0 <proc_info_features+0x3c0>
  403b0d:	c7 05 c9 49 00 00 28 	movl   $0x28,0x49c9(%rip)        # 4084e0 <proc_info_features+0x3e0>
  403b14:	00 00 00 
  403b17:	48 8d 05 67 17 00 00 	lea    0x1767(%rip),%rax        # 405285 <_IO_stdin_used+0x285>
  403b1e:	48 89 05 c3 49 00 00 	mov    %rax,0x49c3(%rip)        # 4084e8 <proc_info_features+0x3e8>
  403b25:	48 8d 05 66 17 00 00 	lea    0x1766(%rip),%rax        # 405292 <_IO_stdin_used+0x292>
  403b2c:	48 89 05 a5 49 00 00 	mov    %rax,0x49a5(%rip)        # 4084d8 <proc_info_features+0x3d8>
  403b33:	c7 05 bb 49 00 00 29 	movl   $0x29,0x49bb(%rip)        # 4084f8 <proc_info_features+0x3f8>
  403b3a:	00 00 00 
  403b3d:	48 8d 05 5c 17 00 00 	lea    0x175c(%rip),%rax        # 4052a0 <_IO_stdin_used+0x2a0>
  403b44:	48 89 05 b5 49 00 00 	mov    %rax,0x49b5(%rip)        # 408500 <proc_info_features+0x400>
  403b4b:	48 8d 05 5b 17 00 00 	lea    0x175b(%rip),%rax        # 4052ad <_IO_stdin_used+0x2ad>
  403b52:	48 89 05 97 49 00 00 	mov    %rax,0x4997(%rip)        # 4084f0 <proc_info_features+0x3f0>
  403b59:	c7 05 ad 49 00 00 2a 	movl   $0x2a,0x49ad(%rip)        # 408510 <proc_info_features+0x410>
  403b60:	00 00 00 
  403b63:	48 8d 05 51 17 00 00 	lea    0x1751(%rip),%rax        # 4052bb <_IO_stdin_used+0x2bb>
  403b6a:	48 89 05 a7 49 00 00 	mov    %rax,0x49a7(%rip)        # 408518 <proc_info_features+0x418>
  403b71:	48 8d 05 53 17 00 00 	lea    0x1753(%rip),%rax        # 4052cb <_IO_stdin_used+0x2cb>
  403b78:	48 89 05 89 49 00 00 	mov    %rax,0x4989(%rip)        # 408508 <proc_info_features+0x408>
  403b7f:	c7 05 9f 49 00 00 2b 	movl   $0x2b,0x499f(%rip)        # 408528 <proc_info_features+0x428>
  403b86:	00 00 00 
  403b89:	48 8d 05 4c 17 00 00 	lea    0x174c(%rip),%rax        # 4052dc <_IO_stdin_used+0x2dc>
  403b90:	48 89 05 99 49 00 00 	mov    %rax,0x4999(%rip)        # 408530 <proc_info_features+0x430>
  403b97:	48 8d 05 4b 17 00 00 	lea    0x174b(%rip),%rax        # 4052e9 <_IO_stdin_used+0x2e9>
  403b9e:	48 89 05 7b 49 00 00 	mov    %rax,0x497b(%rip)        # 408520 <proc_info_features+0x420>
  403ba5:	c7 05 91 49 00 00 2c 	movl   $0x2c,0x4991(%rip)        # 408540 <proc_info_features+0x440>
  403bac:	00 00 00 
  403baf:	48 8d 05 41 17 00 00 	lea    0x1741(%rip),%rax        # 4052f7 <_IO_stdin_used+0x2f7>
  403bb6:	48 89 05 8b 49 00 00 	mov    %rax,0x498b(%rip)        # 408548 <proc_info_features+0x448>
  403bbd:	48 8d 05 3f 17 00 00 	lea    0x173f(%rip),%rax        # 405303 <_IO_stdin_used+0x303>
  403bc4:	48 89 05 6d 49 00 00 	mov    %rax,0x496d(%rip)        # 408538 <proc_info_features+0x438>
  403bcb:	c7 05 83 49 00 00 2d 	movl   $0x2d,0x4983(%rip)        # 408558 <proc_info_features+0x458>
  403bd2:	00 00 00 
  403bd5:	48 8d 05 34 17 00 00 	lea    0x1734(%rip),%rax        # 405310 <_IO_stdin_used+0x310>
  403bdc:	48 89 05 7d 49 00 00 	mov    %rax,0x497d(%rip)        # 408560 <proc_info_features+0x460>
  403be3:	48 8d 05 2b 17 00 00 	lea    0x172b(%rip),%rax        # 405315 <_IO_stdin_used+0x315>
  403bea:	48 89 05 5f 49 00 00 	mov    %rax,0x495f(%rip)        # 408550 <proc_info_features+0x450>
  403bf1:	c7 05 75 49 00 00 2e 	movl   $0x2e,0x4975(%rip)        # 408570 <proc_info_features+0x470>
  403bf8:	00 00 00 
  403bfb:	48 8d 05 18 17 00 00 	lea    0x1718(%rip),%rax        # 40531a <_IO_stdin_used+0x31a>
  403c02:	48 89 05 6f 49 00 00 	mov    %rax,0x496f(%rip)        # 408578 <proc_info_features+0x478>
  403c09:	48 8d 05 0f 17 00 00 	lea    0x170f(%rip),%rax        # 40531f <_IO_stdin_used+0x31f>
  403c10:	48 89 05 51 49 00 00 	mov    %rax,0x4951(%rip)        # 408568 <proc_info_features+0x468>
  403c17:	c7 05 67 49 00 00 2f 	movl   $0x2f,0x4967(%rip)        # 408588 <proc_info_features+0x488>
  403c1e:	00 00 00 
  403c21:	48 8d 05 fc 16 00 00 	lea    0x16fc(%rip),%rax        # 405324 <_IO_stdin_used+0x324>
  403c28:	48 89 05 61 49 00 00 	mov    %rax,0x4961(%rip)        # 408590 <proc_info_features+0x490>
  403c2f:	48 8d 05 f9 16 00 00 	lea    0x16f9(%rip),%rax        # 40532f <_IO_stdin_used+0x32f>
  403c36:	48 89 05 43 49 00 00 	mov    %rax,0x4943(%rip)        # 408580 <proc_info_features+0x480>
  403c3d:	c7 05 59 49 00 00 30 	movl   $0x30,0x4959(%rip)        # 4085a0 <proc_info_features+0x4a0>
  403c44:	00 00 00 
  403c47:	48 8d 05 ec 16 00 00 	lea    0x16ec(%rip),%rax        # 40533a <_IO_stdin_used+0x33a>
  403c4e:	48 89 05 53 49 00 00 	mov    %rax,0x4953(%rip)        # 4085a8 <proc_info_features+0x4a8>
  403c55:	48 8d 05 e9 16 00 00 	lea    0x16e9(%rip),%rax        # 405345 <_IO_stdin_used+0x345>
  403c5c:	48 89 05 35 49 00 00 	mov    %rax,0x4935(%rip)        # 408598 <proc_info_features+0x498>
  403c63:	c7 05 4b 49 00 00 31 	movl   $0x31,0x494b(%rip)        # 4085b8 <proc_info_features+0x4b8>
  403c6a:	00 00 00 
  403c6d:	48 8d 05 dd 16 00 00 	lea    0x16dd(%rip),%rax        # 405351 <_IO_stdin_used+0x351>
  403c74:	48 89 05 45 49 00 00 	mov    %rax,0x4945(%rip)        # 4085c0 <proc_info_features+0x4c0>
  403c7b:	48 8d 05 d4 16 00 00 	lea    0x16d4(%rip),%rax        # 405356 <_IO_stdin_used+0x356>
  403c82:	48 89 05 27 49 00 00 	mov    %rax,0x4927(%rip)        # 4085b0 <proc_info_features+0x4b0>
  403c89:	c7 05 3d 49 00 00 32 	movl   $0x32,0x493d(%rip)        # 4085d0 <proc_info_features+0x4d0>
  403c90:	00 00 00 
  403c93:	48 8d 05 c1 16 00 00 	lea    0x16c1(%rip),%rax        # 40535b <_IO_stdin_used+0x35b>
  403c9a:	48 89 05 37 49 00 00 	mov    %rax,0x4937(%rip)        # 4085d8 <proc_info_features+0x4d8>
  403ca1:	48 8d 05 b9 16 00 00 	lea    0x16b9(%rip),%rax        # 405361 <_IO_stdin_used+0x361>
  403ca8:	48 89 05 19 49 00 00 	mov    %rax,0x4919(%rip)        # 4085c8 <proc_info_features+0x4c8>
  403caf:	c7 05 2f 49 00 00 33 	movl   $0x33,0x492f(%rip)        # 4085e8 <proc_info_features+0x4e8>
  403cb6:	00 00 00 
  403cb9:	48 8d 05 a7 16 00 00 	lea    0x16a7(%rip),%rax        # 405367 <_IO_stdin_used+0x367>
  403cc0:	48 89 05 29 49 00 00 	mov    %rax,0x4929(%rip)        # 4085f0 <proc_info_features+0x4f0>
  403cc7:	48 8d 05 9d 16 00 00 	lea    0x169d(%rip),%rax        # 40536b <_IO_stdin_used+0x36b>
  403cce:	48 89 05 0b 49 00 00 	mov    %rax,0x490b(%rip)        # 4085e0 <proc_info_features+0x4e0>
  403cd5:	c7 05 21 49 00 00 34 	movl   $0x34,0x4921(%rip)        # 408600 <proc_info_features+0x500>
  403cdc:	00 00 00 
  403cdf:	48 8d 05 89 16 00 00 	lea    0x1689(%rip),%rax        # 40536f <_IO_stdin_used+0x36f>
  403ce6:	48 89 05 1b 49 00 00 	mov    %rax,0x491b(%rip)        # 408608 <proc_info_features+0x508>
  403ced:	48 8d 05 81 16 00 00 	lea    0x1681(%rip),%rax        # 405375 <_IO_stdin_used+0x375>
  403cf4:	48 89 05 fd 48 00 00 	mov    %rax,0x48fd(%rip)        # 4085f8 <proc_info_features+0x4f8>
  403cfb:	c7 05 13 49 00 00 35 	movl   $0x35,0x4913(%rip)        # 408618 <proc_info_features+0x518>
  403d02:	00 00 00 
  403d05:	48 8d 05 6f 16 00 00 	lea    0x166f(%rip),%rax        # 40537b <_IO_stdin_used+0x37b>
  403d0c:	48 89 05 0d 49 00 00 	mov    %rax,0x490d(%rip)        # 408620 <proc_info_features+0x520>
  403d13:	48 8d 05 65 16 00 00 	lea    0x1665(%rip),%rax        # 40537f <_IO_stdin_used+0x37f>
  403d1a:	48 89 05 ef 48 00 00 	mov    %rax,0x48ef(%rip)        # 408610 <proc_info_features+0x510>
  403d21:	c7 05 05 49 00 00 36 	movl   $0x36,0x4905(%rip)        # 408630 <proc_info_features+0x530>
  403d28:	00 00 00 
  403d2b:	48 8d 05 51 16 00 00 	lea    0x1651(%rip),%rax        # 405383 <_IO_stdin_used+0x383>
  403d32:	48 89 05 ff 48 00 00 	mov    %rax,0x48ff(%rip)        # 408638 <proc_info_features+0x538>
  403d39:	48 8d 05 4c 16 00 00 	lea    0x164c(%rip),%rax        # 40538c <_IO_stdin_used+0x38c>
  403d40:	48 89 05 e1 48 00 00 	mov    %rax,0x48e1(%rip)        # 408628 <proc_info_features+0x528>
  403d47:	c7 05 f7 48 00 00 37 	movl   $0x37,0x48f7(%rip)        # 408648 <proc_info_features+0x548>
  403d4e:	00 00 00 
  403d51:	48 8d 05 3d 16 00 00 	lea    0x163d(%rip),%rax        # 405395 <_IO_stdin_used+0x395>
  403d58:	48 89 05 f1 48 00 00 	mov    %rax,0x48f1(%rip)        # 408650 <proc_info_features+0x550>
  403d5f:	48 8d 05 37 16 00 00 	lea    0x1637(%rip),%rax        # 40539d <_IO_stdin_used+0x39d>
  403d66:	48 89 05 d3 48 00 00 	mov    %rax,0x48d3(%rip)        # 408640 <proc_info_features+0x540>
  403d6d:	c7 05 e9 48 00 00 38 	movl   $0x38,0x48e9(%rip)        # 408660 <proc_info_features+0x560>
  403d74:	00 00 00 
  403d77:	48 8d 05 27 16 00 00 	lea    0x1627(%rip),%rax        # 4053a5 <_IO_stdin_used+0x3a5>
  403d7e:	48 89 05 e3 48 00 00 	mov    %rax,0x48e3(%rip)        # 408668 <proc_info_features+0x568>
  403d85:	48 8d 05 2c 16 00 00 	lea    0x162c(%rip),%rax        # 4053b8 <_IO_stdin_used+0x3b8>
  403d8c:	48 89 05 c5 48 00 00 	mov    %rax,0x48c5(%rip)        # 408658 <proc_info_features+0x558>
  403d93:	c7 05 db 48 00 00 3c 	movl   $0x3c,0x48db(%rip)        # 408678 <proc_info_features+0x578>
  403d9a:	00 00 00 
  403d9d:	48 8d 05 28 16 00 00 	lea    0x1628(%rip),%rax        # 4053cc <_IO_stdin_used+0x3cc>
  403da4:	48 89 05 d5 48 00 00 	mov    %rax,0x48d5(%rip)        # 408680 <proc_info_features+0x580>
  403dab:	48 8d 05 25 16 00 00 	lea    0x1625(%rip),%rax        # 4053d7 <_IO_stdin_used+0x3d7>
  403db2:	48 89 05 b7 48 00 00 	mov    %rax,0x48b7(%rip)        # 408670 <proc_info_features+0x570>
  403db9:	c7 05 cd 48 00 00 40 	movl   $0x40,0x48cd(%rip)        # 408690 <proc_info_features+0x590>
  403dc0:	00 00 00 
  403dc3:	48 8d 05 19 16 00 00 	lea    0x1619(%rip),%rax        # 4053e3 <_IO_stdin_used+0x3e3>
  403dca:	48 89 05 c7 48 00 00 	mov    %rax,0x48c7(%rip)        # 408698 <proc_info_features+0x598>
  403dd1:	48 8d 05 14 16 00 00 	lea    0x1614(%rip),%rax        # 4053ec <_IO_stdin_used+0x3ec>
  403dd8:	48 89 05 a9 48 00 00 	mov    %rax,0x48a9(%rip)        # 408688 <proc_info_features+0x588>
  403ddf:	c7 05 bf 48 00 00 41 	movl   $0x41,0x48bf(%rip)        # 4086a8 <proc_info_features+0x5a8>
  403de6:	00 00 00 
  403de9:	48 8d 05 05 16 00 00 	lea    0x1605(%rip),%rax        # 4053f5 <_IO_stdin_used+0x3f5>
  403df0:	48 89 05 b9 48 00 00 	mov    %rax,0x48b9(%rip)        # 4086b0 <proc_info_features+0x5b0>
  403df7:	48 8d 05 ff 15 00 00 	lea    0x15ff(%rip),%rax        # 4053fd <_IO_stdin_used+0x3fd>
  403dfe:	48 89 05 9b 48 00 00 	mov    %rax,0x489b(%rip)        # 4086a0 <proc_info_features+0x5a0>
  403e05:	c7 05 b1 48 00 00 42 	movl   $0x42,0x48b1(%rip)        # 4086c0 <proc_info_features+0x5c0>
  403e0c:	00 00 00 
  403e0f:	48 8d 05 ef 15 00 00 	lea    0x15ef(%rip),%rax        # 405405 <_IO_stdin_used+0x405>
  403e16:	48 89 05 ab 48 00 00 	mov    %rax,0x48ab(%rip)        # 4086c8 <proc_info_features+0x5c8>
  403e1d:	48 8d 05 eb 15 00 00 	lea    0x15eb(%rip),%rax        # 40540f <_IO_stdin_used+0x40f>
  403e24:	48 89 05 8d 48 00 00 	mov    %rax,0x488d(%rip)        # 4086b8 <proc_info_features+0x5b8>
  403e2b:	c7 05 a3 48 00 00 43 	movl   $0x43,0x48a3(%rip)        # 4086d8 <proc_info_features+0x5d8>
  403e32:	00 00 00 
  403e35:	48 8d 05 dd 15 00 00 	lea    0x15dd(%rip),%rax        # 405419 <_IO_stdin_used+0x419>
  403e3c:	48 89 05 9d 48 00 00 	mov    %rax,0x489d(%rip)        # 4086e0 <proc_info_features+0x5e0>
  403e43:	48 8d 05 d7 15 00 00 	lea    0x15d7(%rip),%rax        # 405421 <_IO_stdin_used+0x421>
  403e4a:	48 89 05 7f 48 00 00 	mov    %rax,0x487f(%rip)        # 4086d0 <proc_info_features+0x5d0>
  403e51:	c7 05 95 48 00 00 44 	movl   $0x44,0x4895(%rip)        # 4086f0 <proc_info_features+0x5f0>
  403e58:	00 00 00 
  403e5b:	48 8d 05 c7 15 00 00 	lea    0x15c7(%rip),%rax        # 405429 <_IO_stdin_used+0x429>
  403e62:	48 89 05 8f 48 00 00 	mov    %rax,0x488f(%rip)        # 4086f8 <proc_info_features+0x5f8>
  403e69:	48 8d 05 c4 15 00 00 	lea    0x15c4(%rip),%rax        # 405434 <_IO_stdin_used+0x434>
  403e70:	48 89 05 71 48 00 00 	mov    %rax,0x4871(%rip)        # 4086e8 <proc_info_features+0x5e8>
  403e77:	c7 05 87 48 00 00 45 	movl   $0x45,0x4887(%rip)        # 408708 <proc_info_features+0x608>
  403e7e:	00 00 00 
  403e81:	48 8d 05 b8 15 00 00 	lea    0x15b8(%rip),%rax        # 405440 <_IO_stdin_used+0x440>
  403e88:	48 89 05 81 48 00 00 	mov    %rax,0x4881(%rip)        # 408710 <proc_info_features+0x610>
  403e8f:	48 8d 05 b1 15 00 00 	lea    0x15b1(%rip),%rax        # 405447 <_IO_stdin_used+0x447>
  403e96:	48 89 05 63 48 00 00 	mov    %rax,0x4863(%rip)        # 408700 <proc_info_features+0x600>
  403e9d:	c7 05 79 48 00 00 46 	movl   $0x46,0x4879(%rip)        # 408720 <proc_info_features+0x620>
  403ea4:	00 00 00 
  403ea7:	48 8d 05 a0 15 00 00 	lea    0x15a0(%rip),%rax        # 40544e <_IO_stdin_used+0x44e>
  403eae:	48 89 05 73 48 00 00 	mov    %rax,0x4873(%rip)        # 408728 <proc_info_features+0x628>
  403eb5:	48 8d 05 9a 15 00 00 	lea    0x159a(%rip),%rax        # 405456 <_IO_stdin_used+0x456>
  403ebc:	48 89 05 55 48 00 00 	mov    %rax,0x4855(%rip)        # 408718 <proc_info_features+0x618>
  403ec3:	c7 05 6b 48 00 00 47 	movl   $0x47,0x486b(%rip)        # 408738 <proc_info_features+0x638>
  403eca:	00 00 00 
  403ecd:	48 8d 05 8b 15 00 00 	lea    0x158b(%rip),%rax        # 40545f <_IO_stdin_used+0x45f>
  403ed4:	48 89 05 65 48 00 00 	mov    %rax,0x4865(%rip)        # 408740 <proc_info_features+0x640>
  403edb:	48 8d 05 86 15 00 00 	lea    0x1586(%rip),%rax        # 405468 <_IO_stdin_used+0x468>
  403ee2:	48 89 05 47 48 00 00 	mov    %rax,0x4847(%rip)        # 408730 <proc_info_features+0x630>
  403ee9:	c7 05 5d 48 00 00 48 	movl   $0x48,0x485d(%rip)        # 408750 <proc_info_features+0x650>
  403ef0:	00 00 00 
  403ef3:	48 8d 05 77 15 00 00 	lea    0x1577(%rip),%rax        # 405471 <_IO_stdin_used+0x471>
  403efa:	48 89 05 57 48 00 00 	mov    %rax,0x4857(%rip)        # 408758 <proc_info_features+0x658>
  403f01:	48 8d 05 72 15 00 00 	lea    0x1572(%rip),%rax        # 40547a <_IO_stdin_used+0x47a>
  403f08:	48 89 05 39 48 00 00 	mov    %rax,0x4839(%rip)        # 408748 <proc_info_features+0x648>
  403f0f:	c7 05 4f 48 00 00 49 	movl   $0x49,0x484f(%rip)        # 408768 <proc_info_features+0x668>
  403f16:	00 00 00 
  403f19:	48 8d 05 63 15 00 00 	lea    0x1563(%rip),%rax        # 405483 <_IO_stdin_used+0x483>
  403f20:	48 89 05 49 48 00 00 	mov    %rax,0x4849(%rip)        # 408770 <proc_info_features+0x670>
  403f27:	48 8d 05 5e 15 00 00 	lea    0x155e(%rip),%rax        # 40548c <_IO_stdin_used+0x48c>
  403f2e:	48 89 05 2b 48 00 00 	mov    %rax,0x482b(%rip)        # 408760 <proc_info_features+0x660>
  403f35:	c7 05 41 48 00 00 4a 	movl   $0x4a,0x4841(%rip)        # 408780 <proc_info_features+0x680>
  403f3c:	00 00 00 
  403f3f:	48 8d 05 54 15 00 00 	lea    0x1554(%rip),%rax        # 40549a <_IO_stdin_used+0x49a>
  403f46:	48 89 05 3b 48 00 00 	mov    %rax,0x483b(%rip)        # 408788 <proc_info_features+0x688>
  403f4d:	48 8d 05 4e 15 00 00 	lea    0x154e(%rip),%rax        # 4054a2 <_IO_stdin_used+0x4a2>
  403f54:	48 89 05 1d 48 00 00 	mov    %rax,0x481d(%rip)        # 408778 <proc_info_features+0x678>
  403f5b:	c7 05 33 48 00 00 4b 	movl   $0x4b,0x4833(%rip)        # 408798 <proc_info_features+0x698>
  403f62:	00 00 00 
  403f65:	48 8d 05 29 15 00 00 	lea    0x1529(%rip),%rax        # 405495 <_IO_stdin_used+0x495>
  403f6c:	48 89 05 2d 48 00 00 	mov    %rax,0x482d(%rip)        # 4087a0 <proc_info_features+0x6a0>
  403f73:	48 8d 05 23 15 00 00 	lea    0x1523(%rip),%rax        # 40549d <_IO_stdin_used+0x49d>
  403f7a:	48 89 05 0f 48 00 00 	mov    %rax,0x480f(%rip)        # 408790 <proc_info_features+0x690>
  403f81:	c6 05 68 41 00 00 01 	movb   $0x1,0x4168(%rip)        # 4080f0 <__libirc_isa_info_initialized>
  403f88:	59                   	pop    %rcx
  403f89:	c3                   	ret
  403f8a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000403f90 <__libirc_get_feature_bitpos>:
  403f90:	f3 0f 1e fa          	endbr64
  403f94:	51                   	push   %rcx
  403f95:	89 c1                	mov    %eax,%ecx
  403f97:	80 3d 52 41 00 00 00 	cmpb   $0x0,0x4152(%rip)        # 4080f0 <__libirc_isa_info_initialized>
  403f9e:	75 05                	jne    403fa5 <__libirc_get_feature_bitpos+0x15>
  403fa0:	e8 2b f5 ff ff       	call   4034d0 <__libirc_isa_init_once>
  403fa5:	89 c8                	mov    %ecx,%eax
  403fa7:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  403fab:	48 8d 0d 4e 41 00 00 	lea    0x414e(%rip),%rcx        # 408100 <proc_info_features>
  403fb2:	8b 4c c1 08          	mov    0x8(%rcx,%rax,8),%ecx
  403fb6:	8d 41 80             	lea    -0x80(%rcx),%eax
  403fb9:	3d 7f ff ff ff       	cmp    $0xffffff7f,%eax
  403fbe:	b8 fd ff ff ff       	mov    $0xfffffffd,%eax
  403fc3:	0f 43 c1             	cmovae %ecx,%eax
  403fc6:	59                   	pop    %rcx
  403fc7:	c3                   	ret
  403fc8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  403fcf:	00 

0000000000403fd0 <__libirc_get_cpu_feature>:
  403fd0:	f3 0f 1e fa          	endbr64
  403fd4:	50                   	push   %rax
  403fd5:	80 3d 14 41 00 00 00 	cmpb   $0x0,0x4114(%rip)        # 4080f0 <__libirc_isa_info_initialized>
  403fdc:	75 05                	jne    403fe3 <__libirc_get_cpu_feature+0x13>
  403fde:	e8 ed f4 ff ff       	call   4034d0 <__libirc_isa_init_once>
  403fe3:	89 f0                	mov    %esi,%eax
  403fe5:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  403fe9:	48 8d 0d 10 41 00 00 	lea    0x4110(%rip),%rcx        # 408100 <proc_info_features>
  403ff0:	8b 4c c1 08          	mov    0x8(%rcx,%rax,8),%ecx
  403ff4:	8d 41 80             	lea    -0x80(%rcx),%eax
  403ff7:	3d 7f ff ff ff       	cmp    $0xffffff7f,%eax
  403ffc:	b8 fd ff ff ff       	mov    $0xfffffffd,%eax
  404001:	0f 43 c1             	cmovae %ecx,%eax
  404004:	85 c0                	test   %eax,%eax
  404006:	78 14                	js     40401c <__libirc_get_cpu_feature+0x4c>
  404008:	89 c1                	mov    %eax,%ecx
  40400a:	c1 e9 06             	shr    $0x6,%ecx
  40400d:	48 8b 0c cf          	mov    (%rdi,%rcx,8),%rcx
  404011:	31 d2                	xor    %edx,%edx
  404013:	48 0f a3 c1          	bt     %rax,%rcx
  404017:	0f 92 c2             	setb   %dl
  40401a:	89 d0                	mov    %edx,%eax
  40401c:	59                   	pop    %rcx
  40401d:	c3                   	ret
  40401e:	66 90                	xchg   %ax,%ax

0000000000404020 <__libirc_set_cpu_feature>:
  404020:	52                   	push   %rdx
  404021:	56                   	push   %rsi
  404022:	57                   	push   %rdi
  404023:	48 89 c2             	mov    %rax,%rdx
  404026:	80 3d c3 40 00 00 00 	cmpb   $0x0,0x40c3(%rip)        # 4080f0 <__libirc_isa_info_initialized>
  40402d:	75 05                	jne    404034 <__libirc_set_cpu_feature+0x14>
  40402f:	e8 9c f4 ff ff       	call   4034d0 <__libirc_isa_init_once>
  404034:	89 c8                	mov    %ecx,%eax
  404036:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  40403a:	48 8d 0d bf 40 00 00 	lea    0x40bf(%rip),%rcx        # 408100 <proc_info_features>
  404041:	8b 4c c1 08          	mov    0x8(%rcx,%rax,8),%ecx
  404045:	8d 41 80             	lea    -0x80(%rcx),%eax
  404048:	3d 7f ff ff ff       	cmp    $0xffffff7f,%eax
  40404d:	b8 fd ff ff ff       	mov    $0xfffffffd,%eax
  404052:	0f 43 c1             	cmovae %ecx,%eax
  404055:	85 c0                	test   %eax,%eax
  404057:	78 18                	js     404071 <__libirc_set_cpu_feature+0x51>
  404059:	89 c6                	mov    %eax,%esi
  40405b:	c1 ee 06             	shr    $0x6,%esi
  40405e:	83 e0 3f             	and    $0x3f,%eax
  404061:	bf 01 00 00 00       	mov    $0x1,%edi
  404066:	89 c1                	mov    %eax,%ecx
  404068:	48 d3 e7             	shl    %cl,%rdi
  40406b:	48 09 3c f2          	or     %rdi,(%rdx,%rsi,8)
  40406f:	31 c0                	xor    %eax,%eax
  404071:	5f                   	pop    %rdi
  404072:	5e                   	pop    %rsi
  404073:	5a                   	pop    %rdx
  404074:	c3                   	ret
  404075:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  40407c:	00 00 00 
  40407f:	90                   	nop

0000000000404080 <__libirc_handle_intel_isa_disable>:
  404080:	55                   	push   %rbp
  404081:	41 57                	push   %r15
  404083:	41 56                	push   %r14
  404085:	41 54                	push   %r12
  404087:	53                   	push   %rbx
  404088:	31 db                	xor    %ebx,%ebx
  40408a:	48 85 ff             	test   %rdi,%rdi
  40408d:	0f 84 4b 01 00 00    	je     4041de <__libirc_handle_intel_isa_disable+0x15e>
  404093:	49 89 fe             	mov    %rdi,%r14
  404096:	48 8d 3d 2a 10 00 00 	lea    0x102a(%rip),%rdi        # 4050c7 <_IO_stdin_used+0xc7>
  40409d:	e8 8e cf ff ff       	call   401030 <getenv@plt>
  4040a2:	48 85 c0             	test   %rax,%rax
  4040a5:	0f 84 33 01 00 00    	je     4041de <__libirc_handle_intel_isa_disable+0x15e>
  4040ab:	48 89 c2             	mov    %rax,%rdx
  4040ae:	0f b6 00             	movzbl (%rax),%eax
  4040b1:	84 c0                	test   %al,%al
  4040b3:	0f 84 25 01 00 00    	je     4041de <__libirc_handle_intel_isa_disable+0x15e>
  4040b9:	31 db                	xor    %ebx,%ebx
  4040bb:	48 8d 35 3e 40 00 00 	lea    0x403e(%rip),%rsi        # 408100 <proc_info_features>
  4040c2:	31 ff                	xor    %edi,%edi
  4040c4:	4c 8d 42 01          	lea    0x1(%rdx),%r8
  4040c8:	41 b9 01 00 00 00    	mov    $0x1,%r9d
  4040ce:	49 29 d1             	sub    %rdx,%r9
  4040d1:	49 89 d2             	mov    %rdx,%r10
  4040d4:	3c 2c                	cmp    $0x2c,%al
  4040d6:	75 1a                	jne    4040f2 <__libirc_handle_intel_isa_disable+0x72>
  4040d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4040df:	00 
  4040e0:	41 0f b6 42 01       	movzbl 0x1(%r10),%eax
  4040e5:	49 ff c2             	inc    %r10
  4040e8:	49 ff c0             	inc    %r8
  4040eb:	49 ff c9             	dec    %r9
  4040ee:	3c 2c                	cmp    $0x2c,%al
  4040f0:	74 ee                	je     4040e0 <__libirc_handle_intel_isa_disable+0x60>
  4040f2:	0f b6 c0             	movzbl %al,%eax
  4040f5:	85 c0                	test   %eax,%eax
  4040f7:	0f 84 e1 00 00 00    	je     4041de <__libirc_handle_intel_isa_disable+0x15e>
  4040fd:	4c 89 c2             	mov    %r8,%rdx
  404100:	48 89 d0             	mov    %rdx,%rax
  404103:	0f b6 0a             	movzbl (%rdx),%ecx
  404106:	48 ff c2             	inc    %rdx
  404109:	83 f9 2c             	cmp    $0x2c,%ecx
  40410c:	74 12                	je     404120 <__libirc_handle_intel_isa_disable+0xa0>
  40410e:	85 c9                	test   %ecx,%ecx
  404110:	74 0e                	je     404120 <__libirc_handle_intel_isa_disable+0xa0>
  404112:	48 89 c7             	mov    %rax,%rdi
  404115:	eb e9                	jmp    404100 <__libirc_handle_intel_isa_disable+0x80>
  404117:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40411e:	00 00 
  404120:	49 89 fb             	mov    %rdi,%r11
  404123:	4d 29 d3             	sub    %r10,%r11
  404126:	48 ff ca             	dec    %rdx
  404129:	49 ff c3             	inc    %r11
  40412c:	75 0c                	jne    40413a <__libirc_handle_intel_isa_disable+0xba>
  40412e:	0f b6 02             	movzbl (%rdx),%eax
  404131:	84 c0                	test   %al,%al
  404133:	75 8f                	jne    4040c4 <__libirc_handle_intel_isa_disable+0x44>
  404135:	e9 a4 00 00 00       	jmp    4041de <__libirc_handle_intel_isa_disable+0x15e>
  40413a:	80 3d af 3f 00 00 00 	cmpb   $0x0,0x3faf(%rip)        # 4080f0 <__libirc_isa_info_initialized>
  404141:	75 05                	jne    404148 <__libirc_handle_intel_isa_disable+0xc8>
  404143:	e8 88 f3 ff ff       	call   4034d0 <__libirc_isa_init_once>
  404148:	4c 89 d8             	mov    %r11,%rax
  40414b:	48 83 e0 fe          	and    $0xfffffffffffffffe,%rax
  40414f:	49 01 f9             	add    %rdi,%r9
  404152:	49 d1 e9             	shr    %r9
  404155:	b9 01 00 00 00       	mov    $0x1,%ecx
  40415a:	eb 14                	jmp    404170 <__libirc_handle_intel_isa_disable+0xf0>
  40415c:	0f 1f 40 00          	nopl   0x0(%rax)
  404160:	43 80 3c 1f 00       	cmpb   $0x0,(%r15,%r11,1)
  404165:	74 5b                	je     4041c2 <__libirc_handle_intel_isa_disable+0x142>
  404167:	48 ff c1             	inc    %rcx
  40416a:	48 83 f9 47          	cmp    $0x47,%rcx
  40416e:	74 be                	je     40412e <__libirc_handle_intel_isa_disable+0xae>
  404170:	4c 8d 3c 49          	lea    (%rcx,%rcx,2),%r15
  404174:	4e 8b 7c fe 10       	mov    0x10(%rsi,%r15,8),%r15
  404179:	4d 85 ff             	test   %r15,%r15
  40417c:	74 e9                	je     404167 <__libirc_handle_intel_isa_disable+0xe7>
  40417e:	49 83 fb 02          	cmp    $0x2,%r11
  404182:	72 2c                	jb     4041b0 <__libirc_handle_intel_isa_disable+0x130>
  404184:	45 31 e4             	xor    %r12d,%r12d
  404187:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40418e:	00 00 
  404190:	43 0f b6 6c 60 ff    	movzbl -0x1(%r8,%r12,2),%ebp
  404196:	43 3a 2c 67          	cmp    (%r15,%r12,2),%bpl
  40419a:	75 cb                	jne    404167 <__libirc_handle_intel_isa_disable+0xe7>
  40419c:	43 0f b6 2c 60       	movzbl (%r8,%r12,2),%ebp
  4041a1:	43 3a 6c 67 01       	cmp    0x1(%r15,%r12,2),%bpl
  4041a6:	75 bf                	jne    404167 <__libirc_handle_intel_isa_disable+0xe7>
  4041a8:	49 ff c4             	inc    %r12
  4041ab:	4d 39 e1             	cmp    %r12,%r9
  4041ae:	75 e0                	jne    404190 <__libirc_handle_intel_isa_disable+0x110>
  4041b0:	4c 39 d8             	cmp    %r11,%rax
  4041b3:	73 ab                	jae    404160 <__libirc_handle_intel_isa_disable+0xe0>
  4041b5:	41 0f b6 2c 02       	movzbl (%r10,%rax,1),%ebp
  4041ba:	41 3a 2c 07          	cmp    (%r15,%rax,1),%bpl
  4041be:	74 a0                	je     404160 <__libirc_handle_intel_isa_disable+0xe0>
  4041c0:	eb a5                	jmp    404167 <__libirc_handle_intel_isa_disable+0xe7>
  4041c2:	83 f9 02             	cmp    $0x2,%ecx
  4041c5:	0f 82 63 ff ff ff    	jb     40412e <__libirc_handle_intel_isa_disable+0xae>
  4041cb:	4c 89 f0             	mov    %r14,%rax
  4041ce:	e8 4d fe ff ff       	call   404020 <__libirc_set_cpu_feature>
  4041d3:	83 f8 01             	cmp    $0x1,%eax
  4041d6:	83 d3 00             	adc    $0x0,%ebx
  4041d9:	e9 50 ff ff ff       	jmp    40412e <__libirc_handle_intel_isa_disable+0xae>
  4041de:	89 d8                	mov    %ebx,%eax
  4041e0:	5b                   	pop    %rbx
  4041e1:	41 5c                	pop    %r12
  4041e3:	41 5e                	pop    %r14
  4041e5:	41 5f                	pop    %r15
  4041e7:	5d                   	pop    %rbp
  4041e8:	c3                   	ret
  4041e9:	0f 1f 00             	nopl   (%rax)
  4041ec:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004041f0 <__libirc_get_msg>:
  4041f0:	f3 0f 1e fa          	endbr64
  4041f4:	53                   	push   %rbx
  4041f5:	48 81 ec d0 00 00 00 	sub    $0xd0,%rsp
  4041fc:	89 f3                	mov    %esi,%ebx
  4041fe:	48 89 54 24 30       	mov    %rdx,0x30(%rsp)
  404203:	48 89 4c 24 38       	mov    %rcx,0x38(%rsp)
  404208:	4c 89 44 24 40       	mov    %r8,0x40(%rsp)
  40420d:	4c 89 4c 24 48       	mov    %r9,0x48(%rsp)
  404212:	84 c0                	test   %al,%al
  404214:	74 37                	je     40424d <__libirc_get_msg+0x5d>
  404216:	0f 29 44 24 50       	movaps %xmm0,0x50(%rsp)
  40421b:	0f 29 4c 24 60       	movaps %xmm1,0x60(%rsp)
  404220:	0f 29 54 24 70       	movaps %xmm2,0x70(%rsp)
  404225:	0f 29 9c 24 80 00 00 	movaps %xmm3,0x80(%rsp)
  40422c:	00 
  40422d:	0f 29 a4 24 90 00 00 	movaps %xmm4,0x90(%rsp)
  404234:	00 
  404235:	0f 29 ac 24 a0 00 00 	movaps %xmm5,0xa0(%rsp)
  40423c:	00 
  40423d:	0f 29 b4 24 b0 00 00 	movaps %xmm6,0xb0(%rsp)
  404244:	00 
  404245:	0f 29 bc 24 c0 00 00 	movaps %xmm7,0xc0(%rsp)
  40424c:	00 
  40424d:	e8 5e 00 00 00       	call   4042b0 <irc_ptr_msg>
  404252:	85 db                	test   %ebx,%ebx
  404254:	7e 4c                	jle    4042a2 <__libirc_get_msg+0xb2>
  404256:	48 8d 4c 24 20       	lea    0x20(%rsp),%rcx
  40425b:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
  404260:	48 8d 8c 24 e0 00 00 	lea    0xe0(%rsp),%rcx
  404267:	00 
  404268:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  40426d:	48 b9 10 00 00 00 30 	movabs $0x3000000010,%rcx
  404274:	00 00 00 
  404277:	48 89 0c 24          	mov    %rcx,(%rsp)
  40427b:	48 8d 1d 2e 45 00 00 	lea    0x452e(%rip),%rbx        # 4087b0 <get_msg_buf>
  404282:	49 89 e1             	mov    %rsp,%r9
  404285:	be 00 02 00 00       	mov    $0x200,%esi
  40428a:	b9 00 02 00 00       	mov    $0x200,%ecx
  40428f:	48 89 df             	mov    %rbx,%rdi
  404292:	ba 01 00 00 00       	mov    $0x1,%edx
  404297:	49 89 c0             	mov    %rax,%r8
  40429a:	e8 61 ce ff ff       	call   401100 <__vsnprintf_chk@plt>
  40429f:	48 89 d8             	mov    %rbx,%rax
  4042a2:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  4042a9:	5b                   	pop    %rbx
  4042aa:	c3                   	ret
  4042ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004042b0 <irc_ptr_msg>:
  4042b0:	41 57                	push   %r15
  4042b2:	41 56                	push   %r14
  4042b4:	41 54                	push   %r12
  4042b6:	53                   	push   %rbx
  4042b7:	48 81 ec 88 00 00 00 	sub    $0x88,%rsp
  4042be:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4042c5:	00 00 
  4042c7:	48 89 84 24 80 00 00 	mov    %rax,0x80(%rsp)
  4042ce:	00 
  4042cf:	85 ff                	test   %edi,%edi
  4042d1:	74 37                	je     40430a <irc_ptr_msg+0x5a>
  4042d3:	89 fb                	mov    %edi,%ebx
  4042d5:	80 3d d4 48 00 00 00 	cmpb   $0x0,0x48d4(%rip)        # 408bb0 <first_msg>
  4042dc:	74 38                	je     404316 <irc_ptr_msg+0x66>
  4042de:	48 63 c3             	movslq %ebx,%rax
  4042e1:	48 c1 e0 04          	shl    $0x4,%rax
  4042e5:	48 8d 0d e4 36 00 00 	lea    0x36e4(%rip),%rcx        # 4079d0 <irc_msgtab>
  4042ec:	48 8b 44 08 08       	mov    0x8(%rax,%rcx,1),%rax
  4042f1:	80 3d bc 48 00 00 01 	cmpb   $0x1,0x48bc(%rip)        # 408bb4 <use_internal_msg>
  4042f8:	0f 85 04 01 00 00    	jne    404402 <irc_ptr_msg+0x152>
  4042fe:	48 8b 3d b3 48 00 00 	mov    0x48b3(%rip),%rdi        # 408bb8 <message_catalog>
  404305:	e9 e9 00 00 00       	jmp    4043f3 <irc_ptr_msg+0x143>
  40430a:	48 8d 05 97 11 00 00 	lea    0x1197(%rip),%rax        # 4054a8 <_IO_stdin_used+0x4a8>
  404311:	e9 ec 00 00 00       	jmp    404402 <irc_ptr_msg+0x152>
  404316:	c6 05 93 48 00 00 01 	movb   $0x1,0x4893(%rip)        # 408bb0 <first_msg>
  40431d:	48 8d 3d 85 11 00 00 	lea    0x1185(%rip),%rdi        # 4054a9 <_IO_stdin_used+0x4a9>
  404324:	31 f6                	xor    %esi,%esi
  404326:	e8 c5 cd ff ff       	call   4010f0 <catopen@plt>
  40432b:	48 89 c7             	mov    %rax,%rdi
  40432e:	48 89 05 83 48 00 00 	mov    %rax,0x4883(%rip)        # 408bb8 <message_catalog>
  404335:	48 83 f8 ff          	cmp    $0xffffffffffffffff,%rax
  404339:	0f 85 9a 00 00 00    	jne    4043d9 <irc_ptr_msg+0x129>
  40433f:	48 8d 3d 6f 11 00 00 	lea    0x116f(%rip),%rdi        # 4054b5 <_IO_stdin_used+0x4b5>
  404346:	e8 e5 cc ff ff       	call   401030 <getenv@plt>
  40434b:	48 85 c0             	test   %rax,%rax
  40434e:	74 78                	je     4043c8 <irc_ptr_msg+0x118>
  404350:	49 89 e6             	mov    %rsp,%r14
  404353:	ba 80 00 00 00       	mov    $0x80,%edx
  404358:	b9 80 00 00 00       	mov    $0x80,%ecx
  40435d:	4c 89 f7             	mov    %r14,%rdi
  404360:	48 89 c6             	mov    %rax,%rsi
  404363:	e8 a8 cd ff ff       	call   401110 <__strncpy_chk@plt>
  404368:	c6 44 24 7f 00       	movb   $0x0,0x7f(%rsp)
  40436d:	4c 89 f7             	mov    %r14,%rdi
  404370:	be 2e 00 00 00       	mov    $0x2e,%esi
  404375:	e8 26 cd ff ff       	call   4010a0 <strchr@plt>
  40437a:	48 85 c0             	test   %rax,%rax
  40437d:	74 49                	je     4043c8 <irc_ptr_msg+0x118>
  40437f:	49 89 c6             	mov    %rax,%r14
  404382:	c6 00 00             	movb   $0x0,(%rax)
  404385:	4c 8d 3d 29 11 00 00 	lea    0x1129(%rip),%r15        # 4054b5 <_IO_stdin_used+0x4b5>
  40438c:	49 89 e4             	mov    %rsp,%r12
  40438f:	4c 89 ff             	mov    %r15,%rdi
  404392:	4c 89 e6             	mov    %r12,%rsi
  404395:	ba 01 00 00 00       	mov    $0x1,%edx
  40439a:	e8 b1 cc ff ff       	call   401050 <setenv@plt>
  40439f:	48 8d 3d 03 11 00 00 	lea    0x1103(%rip),%rdi        # 4054a9 <_IO_stdin_used+0x4a9>
  4043a6:	31 f6                	xor    %esi,%esi
  4043a8:	e8 43 cd ff ff       	call   4010f0 <catopen@plt>
  4043ad:	48 89 05 04 48 00 00 	mov    %rax,0x4804(%rip)        # 408bb8 <message_catalog>
  4043b4:	41 c6 06 2e          	movb   $0x2e,(%r14)
  4043b8:	4c 89 ff             	mov    %r15,%rdi
  4043bb:	4c 89 e6             	mov    %r12,%rsi
  4043be:	ba 01 00 00 00       	mov    $0x1,%edx
  4043c3:	e8 88 cc ff ff       	call   401050 <setenv@plt>
  4043c8:	48 8b 3d e9 47 00 00 	mov    0x47e9(%rip),%rdi        # 408bb8 <message_catalog>
  4043cf:	48 83 ff ff          	cmp    $0xffffffffffffffff,%rdi
  4043d3:	0f 84 05 ff ff ff    	je     4042de <irc_ptr_msg+0x2e>
  4043d9:	c6 05 d4 47 00 00 01 	movb   $0x1,0x47d4(%rip)        # 408bb4 <use_internal_msg>
  4043e0:	48 63 c3             	movslq %ebx,%rax
  4043e3:	48 c1 e0 04          	shl    $0x4,%rax
  4043e7:	48 8d 0d e2 35 00 00 	lea    0x35e2(%rip),%rcx        # 4079d0 <irc_msgtab>
  4043ee:	48 8b 44 08 08       	mov    0x8(%rax,%rcx,1),%rax
  4043f3:	be 01 00 00 00       	mov    $0x1,%esi
  4043f8:	89 da                	mov    %ebx,%edx
  4043fa:	48 89 c1             	mov    %rax,%rcx
  4043fd:	e8 4e cd ff ff       	call   401150 <catgets@plt>
  404402:	64 48 8b 0c 25 28 00 	mov    %fs:0x28,%rcx
  404409:	00 00 
  40440b:	48 3b 8c 24 80 00 00 	cmp    0x80(%rsp),%rcx
  404412:	00 
  404413:	75 0f                	jne    404424 <irc_ptr_msg+0x174>
  404415:	48 81 c4 88 00 00 00 	add    $0x88,%rsp
  40441c:	5b                   	pop    %rbx
  40441d:	41 5c                	pop    %r12
  40441f:	41 5e                	pop    %r14
  404421:	41 5f                	pop    %r15
  404423:	c3                   	ret
  404424:	e8 67 cc ff ff       	call   401090 <__stack_chk_fail@plt>
  404429:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000404430 <__libirc_print>:
  404430:	f3 0f 1e fa          	endbr64
  404434:	55                   	push   %rbp
  404435:	41 56                	push   %r14
  404437:	53                   	push   %rbx
  404438:	48 81 ec d0 00 00 00 	sub    $0xd0,%rsp
  40443f:	89 fb                	mov    %edi,%ebx
  404441:	48 89 4c 24 38       	mov    %rcx,0x38(%rsp)
  404446:	4c 89 44 24 40       	mov    %r8,0x40(%rsp)
  40444b:	4c 89 4c 24 48       	mov    %r9,0x48(%rsp)
  404450:	84 c0                	test   %al,%al
  404452:	74 37                	je     40448b <__libirc_print+0x5b>
  404454:	0f 29 44 24 50       	movaps %xmm0,0x50(%rsp)
  404459:	0f 29 4c 24 60       	movaps %xmm1,0x60(%rsp)
  40445e:	0f 29 54 24 70       	movaps %xmm2,0x70(%rsp)
  404463:	0f 29 9c 24 80 00 00 	movaps %xmm3,0x80(%rsp)
  40446a:	00 
  40446b:	0f 29 a4 24 90 00 00 	movaps %xmm4,0x90(%rsp)
  404472:	00 
  404473:	0f 29 ac 24 a0 00 00 	movaps %xmm5,0xa0(%rsp)
  40447a:	00 
  40447b:	0f 29 b4 24 b0 00 00 	movaps %xmm6,0xb0(%rsp)
  404482:	00 
  404483:	0f 29 bc 24 c0 00 00 	movaps %xmm7,0xc0(%rsp)
  40448a:	00 
  40448b:	85 f6                	test   %esi,%esi
  40448d:	0f 84 80 00 00 00    	je     404513 <__libirc_print+0xe3>
  404493:	89 d5                	mov    %edx,%ebp
  404495:	89 f7                	mov    %esi,%edi
  404497:	e8 14 fe ff ff       	call   4042b0 <irc_ptr_msg>
  40449c:	85 ed                	test   %ebp,%ebp
  40449e:	7e 4c                	jle    4044ec <__libirc_print+0xbc>
  4044a0:	48 8d 4c 24 20       	lea    0x20(%rsp),%rcx
  4044a5:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
  4044aa:	48 8d 8c 24 f0 00 00 	lea    0xf0(%rsp),%rcx
  4044b1:	00 
  4044b2:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  4044b7:	48 b9 18 00 00 00 30 	movabs $0x3000000018,%rcx
  4044be:	00 00 00 
  4044c1:	48 89 0c 24          	mov    %rcx,(%rsp)
  4044c5:	4c 8d 35 e4 44 00 00 	lea    0x44e4(%rip),%r14        # 4089b0 <print_buf>
  4044cc:	49 89 e1             	mov    %rsp,%r9
  4044cf:	be 00 02 00 00       	mov    $0x200,%esi
  4044d4:	b9 00 02 00 00       	mov    $0x200,%ecx
  4044d9:	4c 89 f7             	mov    %r14,%rdi
  4044dc:	ba 01 00 00 00       	mov    $0x1,%edx
  4044e1:	49 89 c0             	mov    %rax,%r8
  4044e4:	e8 17 cc ff ff       	call   401100 <__vsnprintf_chk@plt>
  4044e9:	4c 89 f0             	mov    %r14,%rax
  4044ec:	83 fb 01             	cmp    $0x1,%ebx
  4044ef:	75 4f                	jne    404540 <__libirc_print+0x110>
  4044f1:	48 8b 0d e0 3a 00 00 	mov    0x3ae0(%rip),%rcx        # 407fd8 <stderr@GLIBC_2.2.5-0xe8>
  4044f8:	48 8b 39             	mov    (%rcx),%rdi
  4044fb:	48 8d 15 a3 0f 00 00 	lea    0xfa3(%rip),%rdx        # 4054a5 <_IO_stdin_used+0x4a5>
  404502:	be 01 00 00 00       	mov    $0x1,%esi
  404507:	48 89 c1             	mov    %rax,%rcx
  40450a:	31 c0                	xor    %eax,%eax
  40450c:	e8 5f cc ff ff       	call   401170 <__fprintf_chk@plt>
  404511:	eb 43                	jmp    404556 <__libirc_print+0x126>
  404513:	83 fb 01             	cmp    $0x1,%ebx
  404516:	75 4a                	jne    404562 <__libirc_print+0x132>
  404518:	48 8b 05 b9 3a 00 00 	mov    0x3ab9(%rip),%rax        # 407fd8 <stderr@GLIBC_2.2.5-0xe8>
  40451f:	48 8b 38             	mov    (%rax),%rdi
  404522:	48 8d 15 7e 0f 00 00 	lea    0xf7e(%rip),%rdx        # 4054a7 <_IO_stdin_used+0x4a7>
  404529:	be 01 00 00 00       	mov    $0x1,%esi
  40452e:	31 c0                	xor    %eax,%eax
  404530:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  404537:	5b                   	pop    %rbx
  404538:	41 5e                	pop    %r14
  40453a:	5d                   	pop    %rbp
  40453b:	e9 30 cc ff ff       	jmp    401170 <__fprintf_chk@plt>
  404540:	48 8d 35 5e 0f 00 00 	lea    0xf5e(%rip),%rsi        # 4054a5 <_IO_stdin_used+0x4a5>
  404547:	bf 01 00 00 00       	mov    $0x1,%edi
  40454c:	48 89 c2             	mov    %rax,%rdx
  40454f:	31 c0                	xor    %eax,%eax
  404551:	e8 ca cb ff ff       	call   401120 <__printf_chk@plt>
  404556:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  40455d:	5b                   	pop    %rbx
  40455e:	41 5e                	pop    %r14
  404560:	5d                   	pop    %rbp
  404561:	c3                   	ret
  404562:	48 8d 35 3e 0f 00 00 	lea    0xf3e(%rip),%rsi        # 4054a7 <_IO_stdin_used+0x4a7>
  404569:	bf 01 00 00 00       	mov    $0x1,%edi
  40456e:	31 c0                	xor    %eax,%eax
  404570:	48 81 c4 d0 00 00 00 	add    $0xd0,%rsp
  404577:	5b                   	pop    %rbx
  404578:	41 5e                	pop    %r14
  40457a:	5d                   	pop    %rbp
  40457b:	e9 a0 cb ff ff       	jmp    401120 <__printf_chk@plt>

Disassembly of section .fini:

0000000000404580 <_fini>:
  404580:	48 83 ec 08          	sub    $0x8,%rsp
  404584:	48 83 c4 08          	add    $0x8,%rsp
  404588:	c3                   	ret
