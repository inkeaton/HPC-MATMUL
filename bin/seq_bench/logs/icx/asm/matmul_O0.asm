
bin/seq_bench/icx/matmul_O0:     file format elf64-x86-64


Disassembly of section .init:

0000000000401000 <_init>:
  401000:	48 83 ec 08          	sub    $0x8,%rsp
  401004:	48 8b 05 c5 2f 00 00 	mov    0x2fc5(%rip),%rax        # 403fd0 <__gmon_start__@Base>
  40100b:	48 85 c0             	test   %rax,%rax
  40100e:	74 02                	je     401012 <_init+0x12>
  401010:	ff d0                	call   *%rax
  401012:	48 83 c4 08          	add    $0x8,%rsp
  401016:	c3                   	ret

Disassembly of section .plt:

0000000000401020 <free@plt-0x10>:
  401020:	ff 35 ca 2f 00 00    	push   0x2fca(%rip)        # 403ff0 <_GLOBAL_OFFSET_TABLE_+0x8>
  401026:	ff 25 cc 2f 00 00    	jmp    *0x2fcc(%rip)        # 403ff8 <_GLOBAL_OFFSET_TABLE_+0x10>
  40102c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401030 <free@plt>:
  401030:	ff 25 ca 2f 00 00    	jmp    *0x2fca(%rip)        # 404000 <free@GLIBC_2.2.5>
  401036:	68 00 00 00 00       	push   $0x0
  40103b:	e9 e0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401040 <clock@plt>:
  401040:	ff 25 c2 2f 00 00    	jmp    *0x2fc2(%rip)        # 404008 <clock@GLIBC_2.2.5>
  401046:	68 01 00 00 00       	push   $0x1
  40104b:	e9 d0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401050 <fclose@plt>:
  401050:	ff 25 ba 2f 00 00    	jmp    *0x2fba(%rip)        # 404010 <fclose@GLIBC_2.2.5>
  401056:	68 02 00 00 00       	push   $0x2
  40105b:	e9 c0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401060 <fprintf@plt>:
  401060:	ff 25 b2 2f 00 00    	jmp    *0x2fb2(%rip)        # 404018 <fprintf@GLIBC_2.2.5>
  401066:	68 03 00 00 00       	push   $0x3
  40106b:	e9 b0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401070 <malloc@plt>:
  401070:	ff 25 aa 2f 00 00    	jmp    *0x2faa(%rip)        # 404020 <malloc@GLIBC_2.2.5>
  401076:	68 04 00 00 00       	push   $0x4
  40107b:	e9 a0 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401080 <fopen@plt>:
  401080:	ff 25 a2 2f 00 00    	jmp    *0x2fa2(%rip)        # 404028 <fopen@GLIBC_2.2.5>
  401086:	68 05 00 00 00       	push   $0x5
  40108b:	e9 90 ff ff ff       	jmp    401020 <_init+0x20>

0000000000401090 <perror@plt>:
  401090:	ff 25 9a 2f 00 00    	jmp    *0x2f9a(%rip)        # 404030 <perror@GLIBC_2.2.5>
  401096:	68 06 00 00 00       	push   $0x6
  40109b:	e9 80 ff ff ff       	jmp    401020 <_init+0x20>

Disassembly of section .plt.got:

00000000004010a0 <__cxa_finalize@plt>:
  4010a0:	ff 25 3a 2f 00 00    	jmp    *0x2f3a(%rip)        # 403fe0 <__cxa_finalize@GLIBC_2.2.5>
  4010a6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004010b0 <_start>:
  4010b0:	31 ed                	xor    %ebp,%ebp
  4010b2:	49 89 d1             	mov    %rdx,%r9
  4010b5:	5e                   	pop    %rsi
  4010b6:	48 89 e2             	mov    %rsp,%rdx
  4010b9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4010bd:	50                   	push   %rax
  4010be:	54                   	push   %rsp
  4010bf:	45 31 c0             	xor    %r8d,%r8d
  4010c2:	31 c9                	xor    %ecx,%ecx
  4010c4:	48 c7 c7 a0 11 40 00 	mov    $0x4011a0,%rdi
  4010cb:	ff 15 ef 2e 00 00    	call   *0x2eef(%rip)        # 403fc0 <__libc_start_main@GLIBC_2.34>
  4010d1:	f4                   	hlt
  4010d2:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4010d9:	00 00 00 
  4010dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004010e0 <deregister_tm_clones>:
  4010e0:	48 8d 3d 61 2f 00 00 	lea    0x2f61(%rip),%rdi        # 404048 <__TMC_END__>
  4010e7:	48 8d 05 5a 2f 00 00 	lea    0x2f5a(%rip),%rax        # 404048 <__TMC_END__>
  4010ee:	48 39 f8             	cmp    %rdi,%rax
  4010f1:	74 15                	je     401108 <deregister_tm_clones+0x28>
  4010f3:	48 8b 05 ce 2e 00 00 	mov    0x2ece(%rip),%rax        # 403fc8 <_ITM_deregisterTMCloneTable@Base>
  4010fa:	48 85 c0             	test   %rax,%rax
  4010fd:	74 09                	je     401108 <deregister_tm_clones+0x28>
  4010ff:	ff e0                	jmp    *%rax
  401101:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  401108:	c3                   	ret
  401109:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401110 <register_tm_clones>:
  401110:	48 8d 3d 31 2f 00 00 	lea    0x2f31(%rip),%rdi        # 404048 <__TMC_END__>
  401117:	48 8d 35 2a 2f 00 00 	lea    0x2f2a(%rip),%rsi        # 404048 <__TMC_END__>
  40111e:	48 29 fe             	sub    %rdi,%rsi
  401121:	48 89 f0             	mov    %rsi,%rax
  401124:	48 c1 ee 3f          	shr    $0x3f,%rsi
  401128:	48 c1 f8 03          	sar    $0x3,%rax
  40112c:	48 01 c6             	add    %rax,%rsi
  40112f:	48 d1 fe             	sar    %rsi
  401132:	74 14                	je     401148 <register_tm_clones+0x38>
  401134:	48 8b 05 9d 2e 00 00 	mov    0x2e9d(%rip),%rax        # 403fd8 <_ITM_registerTMCloneTable@Base>
  40113b:	48 85 c0             	test   %rax,%rax
  40113e:	74 08                	je     401148 <register_tm_clones+0x38>
  401140:	ff e0                	jmp    *%rax
  401142:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401148:	c3                   	ret
  401149:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401150 <__do_global_dtors_aux>:
  401150:	f3 0f 1e fa          	endbr64
  401154:	80 3d 0d 2f 00 00 00 	cmpb   $0x0,0x2f0d(%rip)        # 404068 <completed.0>
  40115b:	75 2b                	jne    401188 <__do_global_dtors_aux+0x38>
  40115d:	55                   	push   %rbp
  40115e:	48 83 3d 7a 2e 00 00 	cmpq   $0x0,0x2e7a(%rip)        # 403fe0 <__cxa_finalize@GLIBC_2.2.5>
  401165:	00 
  401166:	48 89 e5             	mov    %rsp,%rbp
  401169:	74 0c                	je     401177 <__do_global_dtors_aux+0x27>
  40116b:	48 8b 3d ce 2e 00 00 	mov    0x2ece(%rip),%rdi        # 404040 <__dso_handle>
  401172:	e8 29 ff ff ff       	call   4010a0 <__cxa_finalize@plt>
  401177:	e8 64 ff ff ff       	call   4010e0 <deregister_tm_clones>
  40117c:	c6 05 e5 2e 00 00 01 	movb   $0x1,0x2ee5(%rip)        # 404068 <completed.0>
  401183:	5d                   	pop    %rbp
  401184:	c3                   	ret
  401185:	0f 1f 00             	nopl   (%rax)
  401188:	c3                   	ret
  401189:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401190 <frame_dummy>:
  401190:	f3 0f 1e fa          	endbr64
  401194:	e9 77 ff ff ff       	jmp    401110 <register_tm_clones>
  401199:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004011a0 <main>:
  4011a0:	55                   	push   %rbp
  4011a1:	48 89 e5             	mov    %rsp,%rbp
  4011a4:	48 83 ec 60          	sub    $0x60,%rsp
  4011a8:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
  4011af:	89 7d f8             	mov    %edi,-0x8(%rbp)
  4011b2:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4011b6:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4011bb:	e8 b0 fe ff ff       	call   401070 <malloc@plt>
  4011c0:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4011c4:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4011c9:	e8 a2 fe ff ff       	call   401070 <malloc@plt>
  4011ce:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  4011d2:	bf 00 c2 eb 0b       	mov    $0xbebc200,%edi
  4011d7:	e8 94 fe ff ff       	call   401070 <malloc@plt>
  4011dc:	48 89 45 d8          	mov    %rax,-0x28(%rbp)
  4011e0:	c7 45 d4 00 00 00 00 	movl   $0x0,-0x2c(%rbp)
  4011e7:	81 7d d4 88 13 00 00 	cmpl   $0x1388,-0x2c(%rbp)
  4011ee:	0f 8d 99 00 00 00    	jge    40128d <main+0xed>
  4011f4:	c7 45 d0 00 00 00 00 	movl   $0x0,-0x30(%rbp)
  4011fb:	81 7d d0 88 13 00 00 	cmpl   $0x1388,-0x30(%rbp)
  401202:	0f 8d 72 00 00 00    	jge    40127a <main+0xda>
  401208:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40120c:	48 63 4d d4          	movslq -0x2c(%rbp),%rcx
  401210:	48 69 c9 40 9c 00 00 	imul   $0x9c40,%rcx,%rcx
  401217:	48 01 c8             	add    %rcx,%rax
  40121a:	48 63 4d d0          	movslq -0x30(%rbp),%rcx
  40121e:	f2 0f 10 05 f2 0d 00 	movsd  0xdf2(%rip),%xmm0        # 402018 <_IO_stdin_used+0x18>
  401225:	00 
  401226:	f2 0f 11 04 c8       	movsd  %xmm0,(%rax,%rcx,8)
  40122b:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40122f:	48 63 4d d4          	movslq -0x2c(%rbp),%rcx
  401233:	48 69 c9 40 9c 00 00 	imul   $0x9c40,%rcx,%rcx
  40123a:	48 01 c8             	add    %rcx,%rax
  40123d:	48 63 4d d0          	movslq -0x30(%rbp),%rcx
  401241:	f2 0f 10 05 c7 0d 00 	movsd  0xdc7(%rip),%xmm0        # 402010 <_IO_stdin_used+0x10>
  401248:	00 
  401249:	f2 0f 11 04 c8       	movsd  %xmm0,(%rax,%rcx,8)
  40124e:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  401252:	48 63 4d d4          	movslq -0x2c(%rbp),%rcx
  401256:	48 69 c9 40 9c 00 00 	imul   $0x9c40,%rcx,%rcx
  40125d:	48 01 c8             	add    %rcx,%rax
  401260:	48 63 4d d0          	movslq -0x30(%rbp),%rcx
  401264:	0f 57 c0             	xorps  %xmm0,%xmm0
  401267:	f2 0f 11 04 c8       	movsd  %xmm0,(%rax,%rcx,8)
  40126c:	8b 45 d0             	mov    -0x30(%rbp),%eax
  40126f:	83 c0 01             	add    $0x1,%eax
  401272:	89 45 d0             	mov    %eax,-0x30(%rbp)
  401275:	e9 81 ff ff ff       	jmp    4011fb <main+0x5b>
  40127a:	e9 00 00 00 00       	jmp    40127f <main+0xdf>
  40127f:	8b 45 d4             	mov    -0x2c(%rbp),%eax
  401282:	83 c0 01             	add    $0x1,%eax
  401285:	89 45 d4             	mov    %eax,-0x2c(%rbp)
  401288:	e9 5a ff ff ff       	jmp    4011e7 <main+0x47>
  40128d:	e8 ae fd ff ff       	call   401040 <clock@plt>
  401292:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
  401296:	c7 45 c4 00 00 00 00 	movl   $0x0,-0x3c(%rbp)
  40129d:	81 7d c4 88 13 00 00 	cmpl   $0x1388,-0x3c(%rbp)
  4012a4:	0f 8d b2 00 00 00    	jge    40135c <main+0x1bc>
  4012aa:	c7 45 c0 00 00 00 00 	movl   $0x0,-0x40(%rbp)
  4012b1:	81 7d c0 88 13 00 00 	cmpl   $0x1388,-0x40(%rbp)
  4012b8:	0f 8d 8b 00 00 00    	jge    401349 <main+0x1a9>
  4012be:	c7 45 bc 00 00 00 00 	movl   $0x0,-0x44(%rbp)
  4012c5:	81 7d bc 88 13 00 00 	cmpl   $0x1388,-0x44(%rbp)
  4012cc:	0f 8d 64 00 00 00    	jge    401336 <main+0x196>
  4012d2:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4012d6:	48 63 4d c4          	movslq -0x3c(%rbp),%rcx
  4012da:	48 69 c9 40 9c 00 00 	imul   $0x9c40,%rcx,%rcx
  4012e1:	48 01 c8             	add    %rcx,%rax
  4012e4:	48 63 4d c0          	movslq -0x40(%rbp),%rcx
  4012e8:	f2 0f 10 04 c8       	movsd  (%rax,%rcx,8),%xmm0
  4012ed:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4012f1:	48 63 4d c0          	movslq -0x40(%rbp),%rcx
  4012f5:	48 69 c9 40 9c 00 00 	imul   $0x9c40,%rcx,%rcx
  4012fc:	48 01 c8             	add    %rcx,%rax
  4012ff:	48 63 4d bc          	movslq -0x44(%rbp),%rcx
  401303:	f2 0f 59 04 c8       	mulsd  (%rax,%rcx,8),%xmm0
  401308:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  40130c:	48 63 4d c4          	movslq -0x3c(%rbp),%rcx
  401310:	48 69 c9 40 9c 00 00 	imul   $0x9c40,%rcx,%rcx
  401317:	48 01 c8             	add    %rcx,%rax
  40131a:	48 63 4d bc          	movslq -0x44(%rbp),%rcx
  40131e:	f2 0f 58 04 c8       	addsd  (%rax,%rcx,8),%xmm0
  401323:	f2 0f 11 04 c8       	movsd  %xmm0,(%rax,%rcx,8)
  401328:	8b 45 bc             	mov    -0x44(%rbp),%eax
  40132b:	83 c0 01             	add    $0x1,%eax
  40132e:	89 45 bc             	mov    %eax,-0x44(%rbp)
  401331:	e9 8f ff ff ff       	jmp    4012c5 <main+0x125>
  401336:	e9 00 00 00 00       	jmp    40133b <main+0x19b>
  40133b:	8b 45 c0             	mov    -0x40(%rbp),%eax
  40133e:	83 c0 01             	add    $0x1,%eax
  401341:	89 45 c0             	mov    %eax,-0x40(%rbp)
  401344:	e9 68 ff ff ff       	jmp    4012b1 <main+0x111>
  401349:	e9 00 00 00 00       	jmp    40134e <main+0x1ae>
  40134e:	8b 45 c4             	mov    -0x3c(%rbp),%eax
  401351:	83 c0 01             	add    $0x1,%eax
  401354:	89 45 c4             	mov    %eax,-0x3c(%rbp)
  401357:	e9 41 ff ff ff       	jmp    40129d <main+0xfd>
  40135c:	e8 df fc ff ff       	call   401040 <clock@plt>
  401361:	48 2b 45 c8          	sub    -0x38(%rbp),%rax
  401365:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
  401369:	f2 48 0f 2a 45 c8    	cvtsi2sdq -0x38(%rbp),%xmm0
  40136f:	f2 0f 10 0d 91 0c 00 	movsd  0xc91(%rip),%xmm1        # 402008 <_IO_stdin_used+0x8>
  401376:	00 
  401377:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  40137b:	f2 0f 11 45 b0       	movsd  %xmm0,-0x50(%rbp)
  401380:	48 8b 3c 25 60 40 40 	mov    0x404060,%rdi
  401387:	00 
  401388:	f2 0f 10 45 b0       	movsd  -0x50(%rbp),%xmm0
  40138d:	48 be 20 20 40 00 00 	movabs $0x402020,%rsi
  401394:	00 00 00 
  401397:	ba 88 13 00 00       	mov    $0x1388,%edx
  40139c:	b0 01                	mov    $0x1,%al
  40139e:	e8 bd fc ff ff       	call   401060 <fprintf@plt>
  4013a3:	48 bf 3b 20 40 00 00 	movabs $0x40203b,%rdi
  4013aa:	00 00 00 
  4013ad:	48 be 47 20 40 00 00 	movabs $0x402047,%rsi
  4013b4:	00 00 00 
  4013b7:	e8 c4 fc ff ff       	call   401080 <fopen@plt>
  4013bc:	48 89 45 a8          	mov    %rax,-0x58(%rbp)
  4013c0:	48 83 7d a8 00       	cmpq   $0x0,-0x58(%rbp)
  4013c5:	0f 85 1b 00 00 00    	jne    4013e6 <main+0x246>
  4013cb:	48 bf 49 20 40 00 00 	movabs $0x402049,%rdi
  4013d2:	00 00 00 
  4013d5:	e8 b6 fc ff ff       	call   401090 <perror@plt>
  4013da:	c7 45 fc 01 00 00 00 	movl   $0x1,-0x4(%rbp)
  4013e1:	e9 ce 00 00 00       	jmp    4014b4 <main+0x314>
  4013e6:	48 8b 7d a8          	mov    -0x58(%rbp),%rdi
  4013ea:	48 be 4f 20 40 00 00 	movabs $0x40204f,%rsi
  4013f1:	00 00 00 
  4013f4:	ba 88 13 00 00       	mov    $0x1388,%edx
  4013f9:	b0 00                	mov    $0x0,%al
  4013fb:	e8 60 fc ff ff       	call   401060 <fprintf@plt>
  401400:	c7 45 a4 00 00 00 00 	movl   $0x0,-0x5c(%rbp)
  401407:	81 7d a4 e8 03 00 00 	cmpl   $0x3e8,-0x5c(%rbp)
  40140e:	0f 8d 75 00 00 00    	jge    401489 <main+0x2e9>
  401414:	c7 45 a0 00 00 00 00 	movl   $0x0,-0x60(%rbp)
  40141b:	81 7d a0 e8 03 00 00 	cmpl   $0x3e8,-0x60(%rbp)
  401422:	0f 8d 3e 00 00 00    	jge    401466 <main+0x2c6>
  401428:	48 8b 7d a8          	mov    -0x58(%rbp),%rdi
  40142c:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  401430:	48 63 4d a4          	movslq -0x5c(%rbp),%rcx
  401434:	48 69 c9 40 9c 00 00 	imul   $0x9c40,%rcx,%rcx
  40143b:	48 01 c8             	add    %rcx,%rax
  40143e:	48 63 4d a0          	movslq -0x60(%rbp),%rcx
  401442:	f2 0f 10 04 c8       	movsd  (%rax,%rcx,8),%xmm0
  401447:	48 be 54 20 40 00 00 	movabs $0x402054,%rsi
  40144e:	00 00 00 
  401451:	b0 01                	mov    $0x1,%al
  401453:	e8 08 fc ff ff       	call   401060 <fprintf@plt>
  401458:	8b 45 a0             	mov    -0x60(%rbp),%eax
  40145b:	83 c0 01             	add    $0x1,%eax
  40145e:	89 45 a0             	mov    %eax,-0x60(%rbp)
  401461:	e9 b5 ff ff ff       	jmp    40141b <main+0x27b>
  401466:	48 8b 7d a8          	mov    -0x58(%rbp),%rdi
  40146a:	48 be 52 20 40 00 00 	movabs $0x402052,%rsi
  401471:	00 00 00 
  401474:	b0 00                	mov    $0x0,%al
  401476:	e8 e5 fb ff ff       	call   401060 <fprintf@plt>
  40147b:	8b 45 a4             	mov    -0x5c(%rbp),%eax
  40147e:	83 c0 01             	add    $0x1,%eax
  401481:	89 45 a4             	mov    %eax,-0x5c(%rbp)
  401484:	e9 7e ff ff ff       	jmp    401407 <main+0x267>
  401489:	48 8b 7d a8          	mov    -0x58(%rbp),%rdi
  40148d:	e8 be fb ff ff       	call   401050 <fclose@plt>
  401492:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  401496:	e8 95 fb ff ff       	call   401030 <free@plt>
  40149b:	48 8b 7d e0          	mov    -0x20(%rbp),%rdi
  40149f:	e8 8c fb ff ff       	call   401030 <free@plt>
  4014a4:	48 8b 7d d8          	mov    -0x28(%rbp),%rdi
  4014a8:	e8 83 fb ff ff       	call   401030 <free@plt>
  4014ad:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
  4014b4:	8b 45 fc             	mov    -0x4(%rbp),%eax
  4014b7:	48 83 c4 60          	add    $0x60,%rsp
  4014bb:	5d                   	pop    %rbp
  4014bc:	c3                   	ret

Disassembly of section .fini:

00000000004014c0 <_fini>:
  4014c0:	48 83 ec 08          	sub    $0x8,%rsp
  4014c4:	48 83 c4 08          	add    $0x8,%rsp
  4014c8:	c3                   	ret
