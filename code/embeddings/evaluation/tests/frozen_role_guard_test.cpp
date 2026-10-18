// SPDX-License-Identifier: MIT
#include "frozen_role_guard.h"
#include <iostream>
#include <functional>
using namespace embedding::evaluation::frozen_inputs;
int main() {
 try {
  require(sha256("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA256 vector");
  const auto root=fs::temp_directory_path()/("frozen-role-test-"+std::to_string(::getpid()));
  require(fs::create_directory(root),"unique fixture root");
  write_new(root/"a","A");write_new(root/"b","B");
  const std::map<std::string,fs::path> roles{{"a",root/"a"},{"b",root/"b"}};
  const auto row=[](const std::string &value,const std::string &key){return sha256(value)+"  "+key+"\n";};
  write_new(root/"valid",row("A","a")+row("B","b"));
  Guard g(root/"valid",roles);g.verify();require(g.bind("a")==fs::absolute(root/"a").string(),"bound role");
  const auto rejects=[&](const std::string &name,const std::string &manifest,const std::string &error) {
    write_new(root/name,manifest);bool rejected=false;
    try {Guard ignored(root/name,roles);}catch(const std::exception &e){rejected=std::string(e.what()).find(error)!=std::string::npos;}
    require(rejected,"missing rejection: "+name);
  };
  rejects("missing",row("A","a"),"entire closed role set");
  rejects("duplicate",row("A","a")+row("A","a"),"duplicate role/path");
  rejects("sha",row("wrong","a")+row("B","b"),"parent SHA differs");
  // Invalid SHA on the first declared payload must not mask a late undeclared
  // role: complete path admission happens before the first payload hash.
  rejects("late-undeclared",row("wrong","a")+row("B","b")+row("X","extra"),"undeclared role");
  fs::create_symlink(root/"a",root/"link");write_new(root/"redirect",row("A","a")+row("B","b"));
  bool redirected=false;try {Guard ignored(root/"redirect",{{"a",root/"link"},{"b",root/"b"}});}catch(const std::exception &e){redirected=std::string(e.what()).find("redirected role")!=std::string::npos;}
  require(redirected,"symlink rejection");
  {std::ofstream changed(root/"b");changed << "changed";}
  bool changed=false;try{g.verify();}catch(const std::exception &e){changed=std::string(e.what()).find("parent changed")!=std::string::npos;}
  require(changed,"post-use byte mutation rejection");
  // Only this test's exclusively-created Linux temporary directory is removed.
  fs::remove_all(root);std::cout << "Frozen role guard checks passed\n";return 0;
 }catch(const std::exception &e){std::cerr << e.what() << '\n';return 1;}
}
