# 登录git仓库方法

方案A 试过2次没成功过。不知道原因，输入token时，仍然报错，建议用方案B。

## **解决方法（推荐 SSH 或 PAT）**

### **方案 A：使用 Personal Access Token**

1. 登录 `shawlq` GitHub → [生成 PAT](https://github.com/settings/tokens)。
2. 权限至少勾选 `repo`。
3. Push 时：

   ```bash
   Username: shawlq
   Password: <你的 token>
   ```

结果：
```bash
(base) c62664@c62664:~/workdir/gitcode/notes$ git push
Username for 'https://github.com': shawlq
Password for 'https://shawlq@github.com': 
remote: Permission to shawlq/notes.git denied to shawlq.
fatal: unable to access 'https://github.com/shawlq/notes.git/': The requested URL returned error: 403
```

### **方案 B：使用 SSH（最方便）**

1. 生成 SSH key：

   ```bash
   ssh-keygen -t ed25519 -C "你的邮箱"
   ```

   默认存放在 `~/.ssh/id_ed25519`。
2. 添加到 ssh-agent：

   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```
3. 把公钥 `~/.ssh/id_ed25519.pub` 添加到 GitHub → Settings → SSH and GPG keys。
4. 切换仓库远程 URL：

   ```bash
   git remote set-url origin git@github.com:shawlq/notes.git
   ```
5. 直接 `git push`，以后不需要输入用户名和密码。

---

**建议**

* 如果你只有这个账户，**PAT 也可以**，但每次 push 可能要输入（可以缓存）。
* 如果你会在同一台电脑用多个 GitHub 账户，**SSH + ~/.ssh/config** 最安全。

