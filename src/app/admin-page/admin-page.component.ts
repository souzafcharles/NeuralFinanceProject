import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-admin-page',
  templateUrl: './admin-page.component.html',
  styleUrls: ['./admin-page.component.less']
})
export class AdminPageComponent implements OnInit{
  adminType: string | undefined

  constructor(private router: Router){}

  ngOnInit(): void {
    this.autenticacao()
    console.log(this.adminType);
  }
  autenticacao() {
    this.adminType = sessionStorage.getItem('userType')
    if (this.adminType === null || this.adminType === "" || this.adminType === undefined) {
      console.log("aaa " + typeof this.adminType)
      this.router.navigate(['/login']);
    }
  }

}
