import { Component, OnInit } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.less']
})
export class LoginComponent implements OnInit{
  loginForm: FormGroup;
  errorLogin: boolean = false;
  errorCadastro: boolean = false;
  exibeCadastro: boolean = false;
  cadastroForm: FormGroup
  constructor(private router: Router){}

  ngOnInit(): void {
    sessionStorage.removeItem('mail')
    sessionStorage.removeItem('userType')

    this.initLoginForm()
    this.initCadastroForm()
  }

  initLoginForm() {
    this.errorLogin = false
    this.errorCadastro = false
    this.loginForm = new FormGroup({
      login: new FormControl('', Validators.required),
      senha: new FormControl('', Validators.required),
    });
  }

  initCadastroForm() {
    this.errorLogin = false
    this.errorCadastro = false
    this.cadastroForm = new FormGroup({
      login: new FormControl('', Validators.required),
      senha: new FormControl('', Validators.required)
    });
  }

  onSubmitLogin() {
    let loginArray: {[key: string]: string[]} = {
      'freeadmin@email.com': ['123456', 'Free Admin'],
      'basicadmin@email.com': ['123456', 'Basic Admin'],
      'premiumadmin@email.com': ['123456', 'Premium Admin'],
      'freeuser@email.com': ['123456', 'Free User'],
      'basicuser@email.com': ['123456', 'Basic User'],
      'premiumuser@email.com': ['123456', 'Premium User'],
    }
      let login = this.loginForm.get('login').value
      let senha = this.loginForm.get('senha').value
    if (loginArray[login] != null && senha == loginArray[login][0]){
      sessionStorage.setItem('mail', login)
      sessionStorage.setItem('userType', loginArray[login][1])

      if (login.includes('freeuser')){
        this.router.navigate(['/plan-free']);

      }else if (login.includes('basicuser')){
        this.router.navigate(['/plan-basic']);

      }else if (login.includes('premiumuser')){
        this.router.navigate(['/plan-premium']);
      
      }else{
        this.router.navigate(['/admin-page']);
      }


    }else{
      this.errorLogin = true
    }
  }

  abrirCadastrar() {
    this.initCadastroForm()
    this.exibeCadastro = true;
  }

  fecharCadastrar() {
    this.initLoginForm()
    this.exibeCadastro = false;
  }

  onSubmitCadastro() {
    let loginArray: {[key: string]: string} = {
      'freeadmin@email.com': '123456',
      'basicadmin@email.com': '123456',
      'premiumadmin@email.com': '123456',
    }
    let login = this.cadastroForm.get('login').value
    let senha = this.cadastroForm.get('senha').value

    if(login != "" && senha != ""){
      if ( typeof loginArray[login] === 'undefined'){//verifica se existe o email cadastrado no array se nao existe cadastra
        console.log('email: ', login)
        console.log('senha: ',senha)
        this.fecharCadastrar()
      }else{
        this.errorCadastro = true 
      }
    }else{
      this.errorCadastro = true 
    }
  }
}
