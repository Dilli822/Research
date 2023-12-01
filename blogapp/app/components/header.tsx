// import './customCSS/custom.css'; 
import Image from 'next/image';
import React from 'react';
import '../custom.css';

import 'bootstrap/dist/css/bootstrap.min.css';

export default function Header() {
    return (
<header className='head sticky-top'>
  <div className='container-fluid mobContainer'> 
<div className=' nav d-flex align-items-center'>
  <input type='checkbox' id='nav-check'/>
  <div className='nav-header'>
    <div className='nav-title'>
      <figure>
        <Image src='https://unsplash.it/500' alt='logo' width={50} height={50} />
      </figure>
    </div>
  </div>
  <div className='nav-btn'>
    <label htmlFor='nav-check'>
      <span></span>
      <span></span>
      <span></span>
    </label>
  </div>
  
  <div className='nav-links'>
    <a href='/'>Home</a>
    <a href='/about'>About</a>
    <a href='/contact'>Contact</a>
  </div>
  </div>
  </div>


</header>

    )}